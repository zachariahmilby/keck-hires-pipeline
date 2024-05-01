import pickle
import warnings
from pathlib import Path
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from astropy.nddata import CCDData
from lmfit import CompositeModel
from lmfit.models import PolynomialModel, RectangleModel, ConstantModel
from lmfit.parameter import Parameters
from scipy.signal import correlate
from sklearn.metrics import r2_score
from sklearn.preprocessing import minmax_scale

from hirespipeline.files import make_directory
from hirespipeline.general import package_directory, vac_to_air
from hirespipeline.graphics import rcparams
from hirespipeline.order_tracing import _OrderBounds


# TODO: implement refinement with other templates


class _WavelengthSolution:

    def __init__(self, master_arc: CCDData, master_flat: CCDData,
                 order_bounds: _OrderBounds):
        self._master_arc = master_arc
        self._master_flat = master_flat
        self._order_bounds = order_bounds
        self._pixels = np.arange(self._order_bounds.lower_bounds.shape[1])
        self._pixel_edges = np.linspace(
            0, self._pixels.shape[0], self._pixels.shape[0] + 1) - 0.5
        self._slit_width = self._get_slit_width()
        self._1d_arc_spectra = self._make_1d_spectra()
        self._order_numbers = self._get_order_numbers()
        self._fit_pixels = self._make_order_lists()
        self._fit_wavelengths = self._make_order_lists()
        self._fit_ions = self._make_order_lists()
        self._wavelength_solution = self._build_initial_solution()

    def _get_slit_width(self):
        slit_width = (self._master_arc.header['slit_width']
                      / self._master_arc.header['spectral_bin_scale'])
        return slit_width

    def _make_1d_spectra(self):
        """
        Make spatial averages of each spectrum to collapse them to one
        dimension. Also normalize them. To account for overlapping orders, I am
        only selecting the middle 4 rows of data.
        """
        print('      Making one-dimensional arc spectra...')
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
    def _load_templates(cross_disperser: str, detector_layout: str):
        template_filepath = Path(
            package_directory, 'anc',
            f'{detector_layout}_arc_templates_{cross_disperser}.pickle')
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

    def _emission_line_model(
            self, center, amplitude,
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

    def _make_emission_line(self, center: int | float,
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

    def _make_1d_template_spectrum(self, line_centers):
        try:
            return minmax_scale(np.sum([self._make_emission_line(line, 1.0)
                                        for line in line_centers], axis=0))
        except TypeError:
            return np.zeros(self._pixels.shape)

    def _make_template_spectra(self, template):
        artificial_arc_spectrum = np.zeros(
            (len(template['orders']), self._pixels.shape[0]))
        for i, p in enumerate(template['line_centers']):
            artificial_arc_spectrum[i] = self._make_1d_template_spectrum(p)
        return artificial_arc_spectrum

    def _cross_correlate_template(self, template):
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

    def _get_order_numbers(self):
        print('      Determining order numbers...')
        template = self._get_sorted_solution_templates()[0]
        order_numbers, _ = self._cross_correlate_template(template)
        return order_numbers

    @staticmethod
    def _fill_missing_orders(
            data, interp_model: PolynomialModel,
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
        df = pd.read_csv(Path(package_directory, 'anc', 'ThAr_line_list.dat'),
                         sep=' ', names=['wavelength', 'ion'])
        known_lines = vac_to_air(
            df['wavelength'].to_numpy() * u.nm).to(u.nm).value
        known_ions = df['ion'].to_numpy()
        ind = np.abs(known_lines - wavelength).argmin()
        return known_ions[ind]

    def _make_order_lists(self) -> list[list]:
        data = []
        for _ in self._order_numbers:
            data.append([])
        return data

    def _find_sets(self, template_centers, template_wavelengths):
        threshold = int(5 * self._slit_width)
        positions = np.where(
            np.diff(template_centers) > threshold)[0] + 1
        centers_set = np.split(template_centers, positions)
        wavelengths_set = np.split(template_wavelengths, positions)
        return centers_set, wavelengths_set

    def _fit_sets(self, centers_set, wavelengths_set, arc_spectrum,
                  template_spectrum, dx: int = 2.5):

        dx = int(dx * self._slit_width)

        centers = []
        wavelengths = []
        ions = []

        for s, wls in zip(centers_set, wavelengths_set):
            left = np.min(s).astype(int) - dx
            right = np.max(s).astype(int) + dx + 1
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

            fit = model.fit(arc_subset, params, x=pixel_subset)

            for peak in range(n_peaks):
                if fit.params[f'peak{peak}_center'].stderr is None:
                    continue
                elif np.isnan(fit.params[f'peak{peak}_center'].stderr):
                    continue
                centers.append(fit.params[f'peak{peak}_center'].value)
                wavelengths.append(wls[peak])
                ions.append(self._identify_ion(wls[peak]))

        return centers, wavelengths, ions

    def _fit_order(self, centers, wavelengths, ions, n_attempts: int = 10
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray]:

        solution_model = PolynomialModel(degree=3)
        n_pixels = self._pixels.size

        centers = np.array(centers)
        wavelengths = np.array(wavelengths)
        ions = np.array(ions)

        empty = (np.array([]), np.array([]), np.array([]), np.zeros(n_pixels),
                 np.zeros(n_pixels),  np.zeros(n_pixels+1),
                 np.zeros(n_pixels+1), np.array([]))

        if centers.size == 0:
            return empty

        for attempt in range(n_attempts):
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
            max_inv_absv_residual = np.max(inv_absv_residual)
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
                params = solution_model.guess(wavelengths, x=centers)
                fit = solution_model.fit(wavelengths, params, x=centers)
                params = solution_model.guess(centers, x=wavelengths)
                inv_fit = solution_model.fit(centers, params, x=wavelengths)
                residuals = inv_fit.residual

                fit_centers = fit.eval(x=self._pixels)
                fit_centers_unc = inv_fit.eval_uncertainty(x=fit_centers)
                fit_edges = fit.eval(x=self._pixel_edges)
                fit_edges_unc = inv_fit.eval_uncertainty(x=fit_edges)

                return (centers, wavelengths, ions, fit_centers,
                        fit_centers_unc, fit_edges, fit_edges_unc, residuals)

        return empty

    # noinspection DuplicatedCode
    def _build_initial_solution(self):
        """
        Use closest-match template to build an initial wavelength solution.
        """
        print('      Building initial wavelength solution from best-match '
              'template...')
        template = self._get_sorted_solution_templates()[0]
        n_orders = self._order_numbers.size

        fit_centers = np.zeros((n_orders, self._pixels.size))
        fit_centers_unc = np.full_like(fit_centers, fill_value=np.nan)
        fit_edges = np.zeros((n_orders, self._pixel_edges.size))
        fit_edges_unc = np.full_like(fit_edges, fill_value=np.nan)
        found_centers = []
        found_wavelengths = []
        found_ions = []
        residuals = []

        _, spectral_offset = self._cross_correlate_template(template)

        for i in range(n_orders):

            row = np.where(
                template['orders'] == self._order_numbers[i])[0]

            # skip first and last order in all cases
            if (i == 0) | (i == n_orders - 1) | (len(row) == 0):
                found_centers.append([])
                found_wavelengths.append([])
                found_ions.append([])
                residuals.append([])
                continue
            else:
                row = row[0]

            template_wavelengths = template['line_wavelengths'][row]
            template_centers = template['line_centers'][row]
            if len(template_centers) == 0:
                found_centers.append([])
                found_wavelengths.append([])
                found_ions.append([])
                residuals.append([])
                continue
            template_centers -= spectral_offset
            template_spectrum = self._make_1d_template_spectrum(
                template_centers)
            centers_set, wavelengths_set = self._find_sets(
                template_centers, template_wavelengths)

            centers, wavelengths, ions = self._fit_sets(
                centers_set, wavelengths_set, self._1d_arc_spectra[i],
                template_spectrum)

            (centers, wavelengths, ions,
             best_fit_centers, best_fit_centers_unc,
             best_fit_edges, best_fit_edges_unc,
             residual) = self._fit_order(centers, wavelengths, ions)

            found_centers.append(centers)
            found_wavelengths.append(wavelengths)
            found_ions.append(ions)
            fit_centers[i] = best_fit_centers
            fit_centers_unc[i] = best_fit_centers_unc
            fit_edges[i] = best_fit_edges
            fit_edges_unc[i] = best_fit_edges_unc
            residuals.append(residual)

        interp_model = PolynomialModel(degree=3)
        fit_centers = self._fill_missing_orders(
            fit_centers, interp_model, mask_boundaries=False)
        fit_edges = self._fill_missing_orders(
            fit_edges, interp_model, mask_boundaries=False)

        solution = {'fit_centers': fit_centers,
                    'fit_centers_unc': fit_centers_unc,
                    'fit_edges': fit_edges,
                    'fit_edges_unc': fit_edges_unc,
                    'used_pixels': found_centers,
                    'used_wavelengths': found_wavelengths,
                    'used_ions': found_ions,
                    'residual': residuals}

        return solution

    @staticmethod
    def _latexify_ions(ions) -> list[str]:
        return [i.replace('Th', r'Th\,').replace('Ar', r'Ar\,') for i in ions]

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        print(f'      Saving quality assurance graphics...')
        with plt.style.context(rcparams):
            n_orders = len(self._order_numbers)
            for i, order in enumerate(self._order_numbers):
                print(f'         {i + 1}/{n_orders}: Order {order}...',
                      end='\r')
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
                residual = self._wavelength_solution['residual'][i]
                fit_unc = self._wavelength_solution['fit_edges_unc'][i]
                for j in range(pixels.size):
                    x = pixels[j]
                    wl = wavelengths[j]
                    ion = ions[j]
                    y = np.nanmax([spec1d[int(x)-3:int(x)+4]])
                    axes[0].annotate(
                        fr'{wl:.4f} {ion}', xy=(x, y), xytext=(0, 15),
                        ha='left', textcoords='offset points', va='center',
                        rotation=90, rotation_mode='anchor',
                        transform_rotates_text=True, arrowprops=arrowprops,
                        fontsize=3)
                axes[1].scatter(pixels + 0.5, wavelengths, color='grey', s=4)
                axes[1].plot(self._pixels + 0.5, best_fit, color='red',
                             linewidth=0.5)

                axes[2].axhline(0, linestyle='--', color='grey')
                axes[2].scatter(pixels, residual, color='k', s=4)
                axes[2].fill_between(
                    self._pixel_edges + 0.5, fit_unc, -fit_unc,
                    color='grey', alpha=0.5)
                axes[2].annotate('Fit uncertainty',
                                 xy=(0, 1), xytext=(3, -3),
                                 xycoords='axes fraction',
                                 textcoords='offset points',
                                 color='grey', ha='left', va='top')
                if pixels.size == 0:
                    axes[1].annotate('Solution interpolated from other orders',
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
