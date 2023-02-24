import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.modeling import models, fitting
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyUserWarning
from lmfit.models import GaussianModel, PolynomialModel
from scipy.signal import correlate
from sklearn.preprocessing import minmax_scale

from hirespipeline.files import make_directory
from hirespipeline.general import package_directory
from hirespipeline.graphics import rcparams
from hirespipeline.order_tracing import _OrderBounds


class _WavelengthSolution:

    def __init__(self, master_arc: CCDData, master_flat: CCDData,
                 order_bounds: _OrderBounds):
        self._master_arc = master_arc
        self._master_flat = master_flat
        self._order_bounds = order_bounds
        self._pixels = np.arange(self._order_bounds.lower_bounds.shape[1])
        self._pixel_edges = np.linspace(0, self._pixels.shape[0],
                                        self._pixels.shape[0] + 1) - 0.5
        self._slit_half_width = self._get_slit_half_width()
        self._1d_arc_spectra = self._make_1d_spectra()
        self._order_numbers, self._spectral_offset = \
            self._identify_order_numbers()
        self._wavelength_solution = self._calculate_wavelength_solution()

    def _get_slit_half_width(self):
        slit_width = (self._master_arc.header['slit_width']
                      / self._master_arc.header['spectral_bin_scale'])
        return np.ceil(slit_width / 2).astype(int)

    def _make_1d_spectra(self):
        """
        Make spatial averages of each spectrum to collapse them to one
        dimension. Also normalize them.
        """
        spectra = np.full(self._order_bounds.lower_bounds.shape,
                          fill_value=np.nan)
        rectified_arcs = self._order_bounds.rectify(self._master_arc).data
        for i, spectrum in enumerate(rectified_arcs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectra[i] = minmax_scale(np.nanmean(spectrum, axis=0))
        spectra[np.isnan(spectra)] = 0
        return spectra

    @staticmethod
    def _load_templates(cross_disperser: str = 'red'):
        template_filepath = Path(
            package_directory, 'anc',
            f'arc_templates_{cross_disperser}.pickle')
        return pickle.load(open(template_filepath, 'rb'))

    @staticmethod
    def _find_shortest_angle_distances(
            echelle_angle: float, echelle_angles: np.ndarray,
            cross_disperser_angle: float,
            cross_disperser_angles: np.ndarray) -> np.ndarray:
        """
        Return the indices of templates sorted by how close they are to the
        given echelle/cross-disperser angle pairs.
        """
        distances = np.sqrt(
            (echelle_angles - echelle_angle) ** 2 +
            (cross_disperser_angles - cross_disperser_angle) ** 2)
        return np.argsort(distances)

    def _get_closest_solution_templates(self) -> [dict]:
        """
        Sort the solution templates based on their "closeness" to the echelle
        and cross-disperser angles.
        """
        echelle_angle = self._master_arc.header['echelle_angle']
        cross_disperser_angle = \
            self._master_arc.header['cross_disperser_angle']
        cross_disperser = self._master_arc.header['cross_disperser']
        templates = self._load_templates(cross_disperser=cross_disperser)
        echelle_angles = np.array([template['echelle_angle']
                                   for template in templates])
        cross_disperser_angles = np.array([template['cross_disperser_angle']
                                           for template in templates])
        inds = self._find_shortest_angle_distances(
            echelle_angle, echelle_angles,
            cross_disperser_angle, cross_disperser_angles)
        sorted_templates = [templates[ind] for ind in inds]
        return sorted_templates

    def _make_gaussian_emission_line(self, center: int | float) -> np.array:
        """
        Make a normalized Gaussian emission line at a given pixel position with
        a sigma equal to half of the slit width. This evaluates the emission
        line over the entire range of detector spectral pixels, so you can make
        a spectrum of lines by adding them up.
        """
        model = GaussianModel()
        params = model.make_params(amplitude=1, center=center,
                                   sigma=self._slit_half_width)
        return model.eval(x=self._pixels, params=params)

    def _make_1d_template_spectrum(self, line_centers):
        try:
            return minmax_scale(np.sum([self._make_gaussian_emission_line(line)
                                        for line in line_centers], axis=0))
        except TypeError:
            return np.zeros(self._pixels.shape)

    def _make_template_spectra(self, template):
        artificial_arc_spectrum = np.zeros(
            (len(template['orders']), self._pixels.shape[0]))
        for i, p in enumerate(template['line_centers']):
            artificial_arc_spectrum[i] = self._make_1d_template_spectrum(p)
        return artificial_arc_spectrum

    def _identify_order_numbers(self) -> (np.ndarray, int):
        """
        Cross-correlate closest template with the one-dimensional spectra to
        identify the echelle order numbers and determine the offset between the
        master arc and template.
        """
        template = self._get_closest_solution_templates()[0]
        template_spectra = self._make_template_spectra(template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cross_correlation_matrix = correlate(
                self._1d_arc_spectra, template_spectra, mode='same')
        ind = np.unravel_index(cross_correlation_matrix.argmax(),
                               cross_correlation_matrix.shape)
        spatial_offset = int(template_spectra.shape[0] / 2 - ind[0])
        spectral_offset = int(template_spectra.shape[1] / 2 - ind[1])
        order0 = template['orders'][0] - spatial_offset
        orders = np.arange(order0, order0 - self._1d_arc_spectra.shape[0], -1)
        return orders, spectral_offset

    def _calculate_wavelength_solution(self) -> dict:
        template = self._get_closest_solution_templates()[0]
        template_x, template_y = np.meshgrid(self._pixels,
                                             template['orders'])
        xc, yc = np.meshgrid(self._pixels, self._order_numbers)
        xc += self._spectral_offset
        xe, ye = np.meshgrid(self._pixel_edges, self._order_numbers)
        xe += self._spectral_offset

        p_init = models.Polynomial2D(degree=5)
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='Model is linear in parameters',
                                    category=AstropyUserWarning)
            p = fit_p(p_init, template_x, template_y,
                      template['wavelength_solution'])

        fit_centers = np.round(p(xc, yc), 4)
        fit_edges = np.round(p(xe, ye), 4)

        # use solution as starting points and refit for known line locations
        new_fit_centers = np.zeros_like(fit_centers)
        new_fit_edges = np.zeros_like(fit_edges)
        used_pixels = []
        used_wavelengths = []
        known_lines = np.genfromtxt(
            Path(package_directory, 'anc', 'ThAr_line_list.dat'),
            skip_header=True)
        peak_model = GaussianModel()
        solution_model = PolynomialModel(degree=3)
        for i in range(fit_centers.shape[0]):
            fit_order = fit_centers[i]
            arc_order = self._1d_arc_spectra[i]
            lines_in_order = known_lines[
                np.where((known_lines >= np.min(fit_order))
                         & (known_lines <= np.max(fit_order)))]
            new_centers = []
            new_lines = []
            for line in lines_in_order:
                ind = np.abs(fit_order - line).argmin()
                if (ind < 16) or (ind > fit_order.shape[0] - 16):
                    continue
                else:
                    try:
                        y = arc_order[ind-16:ind+16+1]
                        x = np.arange(-16, 16+1, 1)
                        params = peak_model.guess(y, x=x)
                        fit = peak_model.fit(y, params, x=x)
                        fit_center = fit.params['center'].value
                        if np.abs(fit_center) < 16:
                            new_lines.append(line)
                            new_centers.append(ind + fit_center)
                    except IndexError:
                        continue
            params = solution_model.guess(new_lines, x=new_centers)
            fit = solution_model.fit(new_lines, params, x=new_centers)
            new_fit_centers[i] = fit.eval(x=self._pixels)
            new_fit_edges[i] = fit.eval(x=self._pixel_edges)
            used_pixels.append(new_centers)
            used_wavelengths.append(new_lines)
        return {'fit_centers': new_fit_centers,
                'fit_edges': new_fit_edges,
                'used_pixels': used_pixels,
                'used_wavelengths': used_wavelengths}

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        with plt.style.context(rcparams):
            for i in range(len(self._order_numbers)):
                fig, axes = plt.subplots(2, 1, figsize=(4, 3),
                                         sharex='all', constrained_layout=True,
                                         gridspec_kw={'height_ratios': [1, 4]})
                axes[0].plot(self._pixels + 0.5, self._1d_arc_spectra[i],
                             color='k', linewidth=0.5)
                axes[1].scatter(
                    self._wavelength_solution['used_pixels'][i],
                    self._wavelength_solution['used_wavelengths'][i],
                    color='grey', s=4)
                axes[1].plot(self._pixels + 0.5,
                             self._wavelength_solution['fit_centers'][i],
                             color='red', linewidth=0.5)

                axes[0].set_title('Observed Arc Spectrum')
                axes[0].set_yticks([])
                axes[1].set_title('5th-Degree Polynomial Fit')
                axes[1].set_xlabel('Detector Pixel')
                axes[1].xaxis.set_major_locator(ticker.MultipleLocator(512))
                axes[1].set_xlim(0, self._pixels.shape[0])
                axes[1].set_ylabel('Wavelength [nm]')
                axes[1].yaxis.set_major_locator(
                    ticker.MaxNLocator(integer=True))
                savepath = Path(file_path, 'quality_assurance',
                                'wavelength_solutions',
                                f'order{self._order_numbers[i]}.jpg')
                make_directory(savepath.parent)
                plt.savefig(savepath, dpi=600)
                plt.close(fig)

    @property
    def order_numbers(self) -> np.ndarray:
        return self._order_numbers

    @property
    def centers(self) -> np.ndarray:
        return self._wavelength_solution['fit_centers']

    @property
    def edges(self) -> np.ndarray:
        return self._wavelength_solution['fit_edges']
