import pickle
import warnings
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Polynomial2D
from astropy.nddata import CCDData
from astropy.utils.exceptions import AstropyUserWarning
from lmfit import CompositeModel
from lmfit.models import PolynomialModel, RectangleModel, ConstantModel
from lmfit.parameter import Parameters
from scipy.signal import correlate
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale

from hirespipeline.files import make_directory
from hirespipeline.general import package_directory, vac_to_air, _log
from hirespipeline.graphics import rcparams
from hirespipeline.order_tracing import _OrderBounds


class _WavelengthSolution:
    """
    (Attempt to) calculate wavelength solutions for each identified order.
    """
    def __init__(self,
                 master_arc: CCDData,
                 master_flat: CCDData,
                 order_bounds: _OrderBounds,
                 log_path: Path):
        """
        Parameters
        ----------
        master_arc: CCDData
            Master arc image.
        master_flat: CCDData
            Master flat image.
        order_bounds: _OrderBounds
            Order bounds.
        log_path: Path
            Location of log file.
        """
        self._master_arc = master_arc
        self._master_flat = master_flat
        self._order_bounds = order_bounds
        self._log_path = log_path
        self._pixels = np.arange(self._order_bounds.lower_bounds.shape[1])
        self._pixel_edges = np.linspace(
            0, self._pixels.shape[0], self._pixels.shape[0] + 1) - 0.5
        self._slit_width = self._get_slit_width()
        self._1d_arc_spectra = self._make_1d_spectra()
        self._order_numbers = self._get_order_numbers()
        self._solutions_found = np.zeros_like(self._order_numbers, dtype=bool)
        self._fit_pixels = self._make_order_lists()
        self._fit_wavelengths = self._make_order_lists()
        self._fit_ions = self._make_order_lists()
        self._wavelength_solution = self._calculate_optimal_solution()
        self._complete_solution()

    def _get_slit_width(self) -> float:
        """
        Get width of slit in pixels.
        """
        width = self._master_arc.header['slit_width']
        scale = self._master_arc.header['spectral_bin_scale']
        return width / scale

    def _make_1d_spectra(self) -> np.ndarray:
        """
        Make spatial averages of each spectrum to collapse them to one
        dimension. Also normalize them. To account for overlapping orders, I am
        only selecting the middle 4 rows of data.
        """
        _log(self._log_path, '      Making one-dimensional arc spectra...')
        spectra = np.zeros(self._order_bounds.lower_bounds.shape)
        rectified_arcs = self._order_bounds.rectify(self._master_arc).data
        bottom = int(rectified_arcs.shape[1] / 2 - 2)
        top = int(rectified_arcs.shape[1] / 2 + 2) + 1
        select = np.s_[bottom:top]
        for i, spectrum in enumerate(rectified_arcs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectra[i] = minmax_scale(np.nanmean(spectrum[select], axis=0))
        spectra[np.isnan(spectra)] = 0
        return spectra

    @staticmethod
    def _load_templates(cross_disperser: str,
                        detector_layout: str):
        """
        Load arc templates for a specific cross disperser and detector layout
        (mosaic or legacy).
        """
        template_filepath = Path(
            package_directory, 'anc',
            f'{detector_layout}_arc_templates_{cross_disperser}.pickle')
        return pickle.load(open(template_filepath, 'rb'))

    @staticmethod
    def _find_shortest_angle_distances(echelle_angle: float,
                                       echelle_angles: np.ndarray,
                                       cross_disperser_angle: float,
                                       cross_disperser_angles: np.ndarray
                                       ) -> np.ndarray:
        """
        Return the indices of templates sorted by how close they are to the
        given echelle/cross-disperser angle pairs.
        """
        distances = np.sqrt(
            (echelle_angles - echelle_angle) ** 2 +
            (cross_disperser_angles - cross_disperser_angle) ** 2)
        return np.argsort(distances)

    def _get_sorted_solution_templates(self) -> list[dict[str, dict]]:
        """
        Get sorted solution templates based on their "closeness" to the echelle
        and cross-disperser angles.
        """
        echelle_angle = self._master_arc.header['echelle_angle']
        cross_disperser_angle = \
            self._master_arc.header['cross_disperser_angle']
        cross_disperser = self._master_arc.header['cross_disperser']
        detector_layout = self._master_arc.header['detector_layout']
        templates = self._load_templates(cross_disperser=cross_disperser,
                                         detector_layout=detector_layout)
        echelle_angles = np.array([template['echelle_angle']
                                   for template in templates])
        cross_disperser_angles = np.array([template['cross_disperser_angle']
                                           for template in templates])
        inds = self._find_shortest_angle_distances(
            echelle_angle, echelle_angles,
            cross_disperser_angle, cross_disperser_angles)
        sorted_templates = [templates[ind] for ind in inds]
        return sorted_templates

    def _emission_line_model(self,
                             center: int | float,
                             amplitude: int | float,
                             prefix: str = '') -> (CompositeModel, Parameters):
        """
        Model emission line as a rectangle function the width of the slit.
        """
        half_width = self._slit_width/2
        model = RectangleModel(form='logistic', prefix=f'{prefix}_')
        params = Parameters()
        params.add(f'{prefix}_amplitude', value=amplitude, min=0)
        params.add(f'{prefix}_width', value=self._slit_width, vary=False)
        params.add(f'{prefix}_center', value=center, min=center-half_width,
                   max=center+half_width)
        params.add(f'{prefix}_center1',
                   expr=f'{prefix}_center-{prefix}_width/2')
        params.add(f'{prefix}_center2',
                   expr=f'{prefix}_center+{prefix}_width/2')
        params.add(f'{prefix}_sigma1', value=0.7, vary=False)
        params.add(f'{prefix}_sigma2', expr=f'{prefix}_sigma1')
        return model, params

    def _make_emission_line(self,
                            center: int | float,
                            amplitude: int | float) -> np.array:
        """
        Make a normalized Gaussian emission line at a given pixel position with
        a sigma equal to half of the slit width. This evaluates the emission
        line over the entire range of detector spectral pixels, so you can make
        a spectrum of lines by adding them up.
        """
        model, params = self._emission_line_model(
            center=center, amplitude=amplitude)
        return model.eval(x=self._pixels, params=params)

    def _make_1d_template_spectrum(self,
                                   line_centers) -> np.array:
        """
        Generate normalized one-dimensional template spectrum.
        """
        try:
            return minmax_scale(np.sum([self._make_emission_line(line, 1.0)
                                        for line in line_centers], axis=0))
        except TypeError:
            return np.zeros(self._pixels.shape)

    def _make_template_spectra(self,
                               template) -> np.ndarray:
        """
        Make stack of normalized template spectra for each order.
        """
        artificial_arc_spectrum = np.zeros(
            (len(template['orders']), self._pixels.shape[0]))
        for i, p in enumerate(template['line_centers']):
            artificial_arc_spectrum[i] = self._make_1d_template_spectrum(p)
        return artificial_arc_spectrum

    def _cross_correlate_template(self,
                                  template) -> tuple[np.ndarray, int]:
        """
        Cross-correlate template spectra with observed arc lamp spectra.
        """
        template_spectra = self._make_template_spectra(template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cross_correlation_matrix = correlate(
                self._1d_arc_spectra, template_spectra, mode='same')
        ind = np.unravel_index(cross_correlation_matrix.argmax(),
                               cross_correlation_matrix.shape)
        spatial_offset = int(np.round(template_spectra.shape[0] / 2 - ind[0]))
        spectral_offset = int(np.round(template_spectra.shape[1] / 2 - ind[1]))
        order0 = template['orders'][0] - spatial_offset
        orders = np.arange(order0, order0 - self._1d_arc_spectra.shape[0], -1)
        return orders, spectral_offset

    def _get_order_numbers(self) -> np.ndarray:
        """
        Get order numbers identified using the cross-correlation algorithm.
        """
        _log(self._log_path, '      Determining order numbers...')
        template = self._get_sorted_solution_templates()[0]
        order_numbers, _ = self._cross_correlate_template(template)
        return order_numbers

    @staticmethod
    def _fill_missing_orders(data,
                             interp_model: PolynomialModel,
                             mask_boundaries: bool = True) -> np.ndarray:
        """
        Interpolate solution from adjacent orders to fill in missing orders.
        """
        n_orders, n_cols = data.shape
        for col in range(n_cols):
            y = data[:, col]
            ind = np.where(y > 0)
            ind2 = np.where(y == 0)
            x = np.arange(n_orders)
            params = interp_model.guess(y[ind], x=x[ind])
            fit = interp_model.fit(y[ind], params, x=x[ind])
            data[ind2, col] = fit.eval(x=x[ind2])
        if mask_boundaries:
            data[0] = 0
            data[-1] = 0
        return data

    @staticmethod
    def _identify_ion(wavelength) -> str:
        """
        Identify ion from templates using wavelength.
        """
        df = pd.read_csv(Path(package_directory, 'anc', 'ThAr_line_list.dat'),
                         sep=' ', names=['wavelength', 'ion'])
        known_lines = vac_to_air(
            df['wavelength'].to_numpy() * u.nm).to(u.nm).value
        known_ions = df['ion'].to_numpy()
        ind = np.abs(known_lines - wavelength).argmin()
        return known_ions[ind]

    def _make_order_lists(self) -> list[list]:
        """
        Make a list of empty lists, one for each order.
        """
        data = []
        for _ in self._order_numbers:
            data.append([])
        return data

    def _find_sets(self,
                   template_centers,
                   template_wavelengths
                   ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Find 'sets' of wavelengths, where one set could include many blended
        lines.
        """
        threshold = int(5 * self._slit_width)
        positions = np.where(
            np.diff(template_centers) > threshold)[0] + 1
        centers_set = np.split(template_centers, positions)
        wavelengths_set = np.split(template_wavelengths, positions)
        return centers_set, wavelengths_set

    def _fit_sets(self,
                  centers_set,
                  wavelengths_set,
                  arc_spectrum,
                  template_spectrum,
                  dx: int = 2.5) -> tuple[list[float], list[float], list[str]]:
        """
        Fit individual components within each set.
        """
        dx = int(dx * self._slit_width)

        centers = []
        wavelengths = []
        ions = []

        for s, wls in zip(centers_set, wavelengths_set):
            if len(s) == 0:
                continue
            left = np.min(s).astype(int) - dx
            right = np.max(s).astype(int) + dx + 1
            if (left < 32) | (right > self._pixels.size-32):
                continue
            s_ = np.s_[left:right]
            pixel_subset = self._pixels[s_]
            arc_subset = arc_spectrum[s_]
            template_subset = template_spectrum[s_]
            try:
                cross_correlation = correlate(
                    arc_subset, template_subset, mode='same')
            except ValueError:
                continue
            ind = cross_correlation.argmax()
            offset = int(np.round(template_subset.shape[0] / 2 - ind))
            centers_offset = s - offset

            n_peaks = len(s)
            models = []
            paramses = []
            for peak in range(n_peaks):
                model, params = self._emission_line_model(
                    center=centers_offset[peak], amplitude=0.5,
                    prefix=f'peak{peak}')
                models.append(model)
                paramses.append(params)
            model = models[0]
            params = paramses[0]
            if n_peaks > 1:
                for m in models[1:]:
                    model += m
                for p in paramses[1:]:
                    params += p
            model += ConstantModel(prefix='const_')
            params.add(f'const_c', value=0.0)

            try:
                fit = model.fit(arc_subset, params, x=pixel_subset)
            except ValueError:
                continue

            for peak in range(n_peaks):
                if fit.params[f'peak{peak}_center'].stderr is None:
                    continue
                elif np.isnan(fit.params[f'peak{peak}_center'].stderr):
                    continue
                centers.append(fit.params[f'peak{peak}_center'].value)
                wavelengths.append(wls[peak])
                ions.append(self._identify_ion(wls[peak]))

        return centers, wavelengths, ions

    def _fit_order(self,
                   centers,
                   wavelengths,
                   ions,
                   n_attempts: int = 10) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray,
                                                  bool]:
        """
        Calculate wavelength solution for an individual order by fitting a
        polynomial.
        """
        solution_model = PolynomialModel(degree=3)
        n_pixels = self._pixels.size

        centers = np.array(centers).astype(float)
        wavelengths = np.array(wavelengths).astype(float)
        ions = np.array(ions)

        empty = (np.array([]), np.array([]), np.array([]),
                 np.full(n_pixels, fill_value=np.nan),
                 np.full(n_pixels+1, fill_value=np.nan),
                 np.array([]), False)

        if centers.size < 10:
            return empty

        for attempt in range(n_attempts):
            if centers.size < 10:
                return empty
            params = solution_model.guess(wavelengths, x=centers)
            fit = solution_model.fit(wavelengths, params, x=centers)
            absv_residual = np.abs(fit.residual)
            max_absv_residual = np.max(absv_residual)
            inv_sort = np.argsort(wavelengths)
            params = solution_model.guess(
                centers[inv_sort], x=wavelengths[inv_sort])
            inv_fit = solution_model.fit(
                centers[inv_sort], params, x=wavelengths[inv_sort])
            inv_absv_residual = np.abs(inv_fit.residual)
            max_inv_absv_residual = np.std(inv_absv_residual)
            rsquared = r2_score(wavelengths, fit.best_fit)

            if rsquared < 0.9999:
                select = np.where(absv_residual < max_absv_residual)[0]
                centers = centers[select]
                wavelengths = wavelengths[select]
                ions = ions[select]
                continue
            elif (rsquared >= 0.9999) & (max_inv_absv_residual > 0.5):
                select = np.where(
                    inv_absv_residual < max_inv_absv_residual)[0]
                centers = centers[select]
                wavelengths = wavelengths[select]
                ions = ions[select]
                continue
            else:
                init_fit = solution_model.fit(
                    wavelengths, params, x=centers)
                init_solution = init_fit.eval(x=self._pixels)
                keep = np.where((wavelengths > init_solution[0]) &
                                (wavelengths < init_solution[-1]))[0]
                centers = centers[keep]
                wavelengths = wavelengths[keep]
                ions = ions[keep]
                if centers.size < 10:
                    return empty
                params = solution_model.guess(wavelengths, x=centers)
                fit = solution_model.fit(wavelengths, params, x=centers)
                params = solution_model.guess(centers, x=wavelengths)
                inv_fit = solution_model.fit(centers, params, x=wavelengths)
                residuals = inv_fit.residual

                fit_centers = fit.eval(x=self._pixels)
                fit_edges = fit.eval(x=self._pixel_edges)

                return (centers, wavelengths, ions, fit_centers, fit_edges,
                        residuals, True)

        return empty

    # noinspection DuplicatedCode
    def _calculate_optimal_solution(self) -> dict:
        """
        Loop through all the templates and determine which one contains the
        most identified lines for each order.
        """
        _log(self._log_path,
             '      Building optimal wavelength solution from templates...')
        templates = self._get_sorted_solution_templates()
        n_templates = len(templates)
        n_orders = self._order_numbers.size
        n_pixels = self._pixels.size
        n_edges = self._pixel_edges.size

        fit_centers = np.zeros((n_orders, n_pixels))
        fit_edges = np.zeros((n_orders, n_edges))

        solutions = {}
        for order in self._order_numbers:
            solutions[f'order{order}'] = {'used_pixels': np.array([]),
                                          'used_wavelengths': np.array([]),
                                          'used_ions': np.array([]),
                                          'residual': np.array([])}

        for n, template in enumerate(templates):

            _log(self._log_path, f'         Template {n+1}/{n_templates}',
                 new_line=False)

            _, spectral_offset = self._cross_correlate_template(template)

            for i, order in enumerate(self._order_numbers):

                row = np.where(template['orders'] == order)[0]
                if len(row) == 0:
                    continue
                else:
                    row = row[0]

                template_pixels = template['line_centers'][row]
                template_wavelengths = template['line_wavelengths'][row]
                if len(template_pixels) == 0:
                    continue
                template_pixels -= spectral_offset

                select = np.where((template_pixels >= 0) &
                                  (template_pixels < n_pixels))[0]

                template_pixels = template_pixels[select]
                template_wavelengths = template_wavelengths[select]

                template_spectrum = self._make_1d_template_spectrum(
                    template_pixels)
                centers_set, wavelengths_set = self._find_sets(
                    template_pixels, template_wavelengths)

                centers, wavelengths, ions = self._fit_sets(
                    centers_set, wavelengths_set, self._1d_arc_spectra[i],
                    template_spectrum)

                (centers, wavelengths, ions, best_fit_centers, best_fit_edges,
                 residual, success) = self._fit_order(
                    centers, wavelengths, ions)

                used_pixels = len(solutions[f'order{order}']['used_pixels'])

                if (len(centers) > used_pixels) & success:
                    solutions[f'order{order}']['used_pixels'] = centers
                    solutions[f'order{order}']['used_wavelengths'] = \
                        wavelengths
                    solutions[f'order{order}']['used_ions'] = ions
                    fit_centers[i] = best_fit_centers
                    fit_edges[i] = best_fit_edges
                    solutions[f'order{order}']['residual'] = residual
                    self._solutions_found[i] = success

        used_pixels = []
        used_wavelengths = []
        used_ions = []
        residuals = []

        for i, order in enumerate(self._order_numbers):
            used_pixels.append(solutions[f'order{order}']['used_pixels'])
            used_wavelengths.append(
                solutions[f'order{order}']['used_wavelengths'])
            used_ions.append(solutions[f'order{order}']['used_ions'])
            residuals.append(solutions[f'order{order}']['residual'])

        solution = {'fit_centers': fit_centers,
                    'fit_edges': fit_edges,
                    'used_pixels': used_pixels,
                    'used_wavelengths': used_wavelengths,
                    'used_ions': used_ions,
                    'residuals': residuals}

        return solution

    def _complete_solution(self) -> None:
        """
        Calculate full wavelength solution for all orders by fitting a 2D
        surface.
        """
        pixel_centers, orders_centers = np.meshgrid(self._pixels,
                                                    self._order_numbers)
        pixel_edges, orders_edges = np.meshgrid(
            self._pixel_edges, self._order_numbers)
        model = Polynomial2D(degree=3)
        fitter = LevMarLSQFitter()

        solution_centers = self._wavelength_solution['fit_centers']
        solution_edges = self._wavelength_solution['fit_edges']
        used_pixels = self._wavelength_solution['used_pixels']
        ind = []
        missing = []
        for i in range(self._order_numbers.size):
            if ((len(used_pixels[i]) < 10) &
                    (~np.isnan(np.sum(solution_centers[i])))):
                missing.append(i)
            else:
                ind.append(i)
        ind = np.array(ind)
        missing = np.array(missing)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='Model is linear in parameters',
                                    category=AstropyUserWarning)
            fit_centers = fitter(model,
                                 pixel_centers[ind],  # noqa
                                 orders_centers[ind],  # noqa
                                 solution_centers[ind])
            fit_edges = fitter(model,
                               pixel_edges[ind],  # noqa
                               orders_edges[ind],  # noqa
                               solution_edges[ind])

        solution_centers = fit_centers(pixel_centers, orders_centers)
        solution_edges = fit_edges(pixel_edges, orders_edges)
        for i in missing:
            centers = solution_centers[i]
            edges = solution_edges[i]
            self._wavelength_solution['fit_centers'][i] = centers
            self._wavelength_solution['fit_edges'][i] = edges
            self._wavelength_solution['used_pixels'][i] = []
            self._wavelength_solution['used_wavelengths'][i] = []
            self._wavelength_solution['residuals'][i] = []

    @staticmethod
    def _latexify_ions(ions) -> list[str]:
        """
        Add a space after atom abbreviation.
        """
        return [i.replace('Th', 'Th ').replace('Ar', 'Ar ') for i in ions]

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        """
        Generate a quality assurance graphic for a wavelength solution fit.
        """
        with plt.style.context(rcparams):
            for i, order in enumerate(self._order_numbers):
                fig, axes = plt.subplots(
                    3, 1, figsize=(4, 5), sharex='all',
                    gridspec_kw={'height_ratios': [3, 3, 1]},
                    constrained_layout=True, clear=True)
                spec1d = self._1d_arc_spectra[i]
                axes[0].plot(self._pixels + 0.5, spec1d, color='k',
                             linewidth=0.5)
                arrowprops = dict(arrowstyle='->', connectionstyle='arc3')
                pixels = np.array(
                    self._wavelength_solution['used_pixels'][i])
                wavelengths = np.array(
                    self._wavelength_solution['used_wavelengths'][i])
                ions = self._latexify_ions(
                    self._wavelength_solution['used_ions'][i])
                best_fit = self._wavelength_solution['fit_centers'][i]
                residual = self._wavelength_solution['residuals'][i]
                for j in range(pixels.size):
                    x = pixels[j]
                    wl = wavelengths[j]
                    ion = ions[j]
                    y = np.nanmax([spec1d[int(x)-3:int(x)+4]])
                    axes[0].annotate(
                        fr'{wl:.4f} nm {ion}', xy=(x, y), xytext=(0, 15),
                        ha='left', textcoords='offset points', va='center',
                        rotation=90, rotation_mode='anchor',
                        transform_rotates_text=True, arrowprops=arrowprops,
                        fontsize=3)
                if pixels.size > 10:
                    axes[1].scatter(pixels+0.5, wavelengths, color='grey', s=4)
                axes[1].plot(self._pixels+0.5, best_fit, color='red',
                             linewidth=0.5)

                axes[2].axhline(0, linestyle='--', color='grey')
                axes[2].scatter(pixels, residual, color='k', s=4)
                if pixels.size <= 10:
                    axes[1].annotate('Fewer than 10 lines identified' + '\n' +
                                     'Solution interpolated from other orders',
                                     xy=(0, 1), xytext=(3, -3),
                                     xycoords='axes fraction',
                                     textcoords='offset points',
                                     color='red', ha='left', va='top')

                n_pixels = self._pixels.size
                axes[0].set_title('Observed Arc Spectrum')
                axes[0].set_yscale('symlog', linthresh=1e-2, linscale=0.25)
                axes[0].set_ylim(bottom=0, top=10)
                axes[0].set_yticks([])
                axes[0].set_yticks([], minor=True)
                axes[1].set_title('3rd-Degree Polynomial Fit')
                axes[2].set_xlabel('Detector Pixel')
                axes[1].xaxis.set_major_locator(
                    ticker.MultipleLocator(n_pixels/8))
                axes[1].xaxis.set_minor_locator(
                    ticker.MultipleLocator(n_pixels/32))
                axes[2].set_xlim(0, self._pixels.shape[0])
                axes[1].set_ylabel('Wavelength [nm]')
                axes[2].set_ylabel('Residual [pix]')
                axes[1].yaxis.set_major_locator(
                    ticker.MaxNLocator(integer=True))
                ylim = np.max(np.abs(axes[2].get_ylim()))
                axes[2].set_ylim(-ylim, ylim)
                axes[2].yaxis.set_minor_locator(ticker.AutoMinorLocator())
                savepath = Path(file_path, 'quality_assurance',
                                'wavelength_solutions',
                                f'order{self._order_numbers[i]}.jpg')
                make_directory(savepath.parent)
                plt.savefig(savepath, dpi=600)
                plt.close(fig)

    @property
    def order_numbers(self) -> np.ndarray:
        """
        Order number for each order identified in the detector image.
        """
        return self._order_numbers

    @property
    def centers(self) -> np.ndarray:
        """
        Best-fit pixel center wavelengths.
        """
        return self._wavelength_solution['fit_centers']

    @property
    def edges(self) -> np.ndarray:
        """
        Best-fit pixel edge wavelengths.
        """
        return self._wavelength_solution['fit_edges']
