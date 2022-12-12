import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyUserWarning
from lmfit.models import GaussianModel
from scipy.signal import correlate
from sklearn.preprocessing import minmax_scale

from hirespipeline.general import package_directory
from hirespipeline.order_tracing import _OrderBounds
from hirespipeline.files import make_directory


class _WavelengthSolution:

    def __init__(self, master_arc: CCDData, master_flat: CCDData,
                 order_bounds: _OrderBounds):
        self._master_arc = master_arc
        self._master_flat = master_flat
        self._order_bounds = order_bounds
        self._pixels = np.arange(self._order_bounds.lower_bounds.shape[1])
        self._pixel_edges = np.linspace(-0.5, self._pixels.shape[0]-0.5,
                                        self._pixels.shape[0]+1)
        self._slit_half_width = self._get_slit_half_width()
        self._1d_arc_spectra = self._make_1d_spectra()
        self._order_numbers, self._spectral_offset = \
            self._identify_order_numbers()
        self._wavelength_solution_centers, self._wavelength_solution_edges = \
            self._calculate_wavelength_solution()

    def _get_slit_half_width(self):
        slit_width = (self._master_arc.header['slit_width']
                      / self._master_arc.header['spectral_bin_scale'])
        return np.ceil(slit_width / 2).astype(int)

    def _make_1d_spectra(self):
        """
        Make spatial averages of each spectrum to collapse them to one
        dimension. Also normalize them.
        """
        spectra = np.zeros(self._order_bounds.lower_bounds.shape, dtype=float)
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

    def _calculate_wavelength_solution(self):
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
        return fit_centers, fit_edges

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        cmap = plt.get_cmap('bone').copy()
        axis.pcolormesh(self._master_flat.data, cmap=cmap, alpha=0.5)
        lb = self._order_bounds.lower_bounds
        ub = self._order_bounds.upper_bounds
        for i in range(len(self._order_numbers)):
            x = 2047
            y = lb[i][x] + (ub[i][x] - lb[i][x]) / 2
            axis.text(x, y, self._order_numbers[i], ha='center',
                      va='center_baseline', fontsize=6, color='red')
            x = 16
            y = lb[i][x] + (ub[i][x] - lb[i][x]) / 2
            wl = f'{self._wavelength_solution_centers[i][0]:.4f} nm'
            axis.text(x, y, wl, ha='left', va='center_baseline', fontsize=6,
                      color='blue')
            x = 4096-16
            y = lb[i][x] + (ub[i][x] - lb[i][x]) / 2
            wl = f'{self._wavelength_solution_centers[i][-1]:.4f} nm'
            axis.text(x, y, wl, ha='right', va='center_baseline', fontsize=6,
                      color='blue')
        savepath = Path(file_path, 'quality_assurance',
                        'order_numbers_and_wavelength_bounds.jpg')
        make_directory(savepath.parent)
        plt.savefig(savepath, dpi=600)
        plt.close(fig)

    @property
    def order_numbers(self) -> np.ndarray:
        return self._order_numbers

    @property
    def centers(self) -> np.ndarray:
        return self._wavelength_solution_centers

    @property
    def edges(self) -> np.ndarray:
        return self._wavelength_solution_edges
