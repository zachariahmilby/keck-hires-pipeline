import pickle
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
from lmfit.models import GaussianModel, PolynomialModel
from scipy.signal import find_peaks, correlate
from sklearn.preprocessing import minmax_scale

from hirespipeline.files import _Files, _make_directory
from hirespipeline.general import package_directory
from hirespipeline.images import _retrieve_ccd_data_from_fits, _CCDData, \
    _remove_outliers


def _remove_header_info_for_masters(header: dict) -> dict:
    header = deepcopy(header)
    header['file_name'] = None
    header['datetime'] = None
    header['exposure_time'] = None
    header['airmass'] = None
    return header


class _MasterBias:
    """
    Make a master bias image by taking the median of a set of bias images.
    Includes a method for subtracting the bias from another _CCDData object
    and logging it.
    """

    def __init__(self, root_directory: Path):
        self._directory = Path(root_directory, 'bias')
        self._master_bias, self._header = self._make_master_bias()

    def _make_master_bias(self) -> (u.Quantity, dict):
        files = _Files(self._directory)
        ccd_data = []
        data = _retrieve_ccd_data_from_fits(files.paths[0])
        unit = data.image.unit
        header = _remove_header_info_for_masters(data.header)
        for file in files.paths:
            ccd_data.append(_retrieve_ccd_data_from_fits(file).image.value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_image = np.nanmedian(np.asarray(ccd_data), axis=0) * unit
        return median_image, header

    def subtract_bias(self, data: _CCDData) -> _CCDData:
        image = deepcopy(data.image)
        uncertainty = deepcopy(data.uncertainty)
        header = deepcopy(data.header)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image -= self._master_bias
        header['processing_log'] += ['subtract_bias']
        return _CCDData(image, uncertainty, header)

    @property
    def image(self) -> np.ndarray:
        return self._master_bias

    @property
    def header(self) -> dict:
        return self._header


class _MasterFlat:
    """
    Make a master flat image by taking the median of a set of flat images, then
    normalize it by the mean. Includes a method for flat-correcting another
    _CCDData object and logging it.
    """

    def __init__(self, root_directory: Path, master_bias: _MasterBias):
        self._directory = Path(root_directory, 'flat')
        self._master_bias = master_bias
        self._master_flat, self._header = self._make_master_flat()

    def _make_master_flat(self) -> (np.ndarray, dict):
        files = _Files(self._directory)
        ccd_data = []
        header = None
        for file in files.paths:
            raw_data = _retrieve_ccd_data_from_fits(file)
            bias_subtracted_data = self._master_bias.subtract_bias(raw_data)
            ccd_data.append(bias_subtracted_data.image.value)
            if header is None:
                header = _remove_header_info_for_masters(
                    bias_subtracted_data.header)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_image = _remove_outliers(
                np.nanmedian(np.asarray(ccd_data), axis=0))
            normalized_image = median_image / np.nanmean(median_image)
        return normalized_image, header

    def flat_correct(self, data: _CCDData) -> _CCDData:
        if 'subtract_bias' not in data.header['processing_log']:
            data = self._master_bias.subtract_bias(data)
        image = deepcopy(data.image)
        uncertainty = deepcopy(data.uncertainty)
        header = deepcopy(data.header)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image /= self._master_flat
        header['processing_log'] += ['flat_correct']
        return _CCDData(image, uncertainty, header)

    @property
    def image(self) -> np.ndarray:
        return self._master_flat

    @property
    def header(self) -> dict:
        return self._header


class _MasterArc:
    """
    Make a master arc image by taking the median of a set of arc images, then
    normalize it by the mean. Also replaces saturated pixels with nan.
    """

    def __init__(self, root_directory: Path, master_bias: _MasterBias,
                 master_flat: _MasterFlat):
        self._directory = Path(root_directory, 'arc')
        self._master_bias = master_bias
        self._master_flat = master_flat
        self._master_arc, self._header = self._make_master_arc()

    def _make_master_arc(self) -> (np.ndarray, dict):
        files = _Files(self._directory)
        ccd_data = []
        header = None
        for file in files.paths:
            raw_data = _retrieve_ccd_data_from_fits(file)
            raw_data.image[raw_data.image == 0] = np.nan
            bias_subtracted_data = self._master_bias.subtract_bias(raw_data)
            flat_corrected_data = self._master_flat.flat_correct(
                bias_subtracted_data)
            ccd_data.append(flat_corrected_data.image.value)
            if header is None:
                header = _remove_header_info_for_masters(
                    flat_corrected_data.header)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_image = _remove_outliers(
                np.nanmedian(np.asarray(ccd_data), axis=0))
            normalized_image = minmax_scale(median_image)
        return normalized_image, header

    @property
    def image(self) -> np.ndarray:
        return self._master_arc

    @property
    def header(self) -> dict:
        return self._header


class _MasterTrace:
    """
    Make a master trace image.
    """

    def __init__(self, root_directory: Path, master_bias: _MasterBias,
                 master_flat: _MasterFlat):
        self._directory = Path(root_directory, 'trace')
        self._master_bias = master_bias
        self._master_flat = master_flat
        self._master_trace, self._header = self._get_trace_image()
        self._n_rows, self._n_cols = self._master_trace.shape
        self._selected_pixels = np.arange(0, self._n_cols, 128, dtype=int)
        self._pixels = np.arange(self._n_cols)
        self._traces = self._find_complete_traces()

    def _get_trace_image(self):
        """
        Make a bias-subtracted, normalized master trace image.
        """
        file = _Files(self._directory).paths[0]
        raw_data = _retrieve_ccd_data_from_fits(file)
        bias_subtracted_data = self._master_bias.subtract_bias(raw_data)
        normalized_image = minmax_scale(bias_subtracted_data.image)
        return normalized_image, raw_data.header

    def _fit_gaussian(self, center, col):
        center = int(center)
        y = self._master_trace[center-15:center+15, col]
        x = np.arange(center-15, center+15)
        model = GaussianModel()
        params = model.guess(y, x=x)
        result = model.fit(y, params, x=x)
        center = result.params['center'].value
        return center

    @staticmethod
    def _fit_polynomial(y: np.ndarray, x: np.ndarray, degree: int):
        model = PolynomialModel(degree=degree)
        ind = np.isfinite(y)
        params = model.guess(y[ind], x=x[ind])
        result = model.fit(y, params, x=x, nan_policy='omit')
        return result

    def _find_initial_trace(self) -> (np.ndarray, int, bool):
        """
        Calculate an initial trace using the third identified peak (should be
        far enough for the whole trace to fall on the detector without any odd
        effects. Also returns the initial spacing between this trace and its
        adjacent traces.
        """
        vslice = self._master_trace[:, 0]
        peaks, _ = find_peaks(vslice, height=0.2)
        use_peak = 0
        center = peaks[use_peak]
        dy = peaks[use_peak+1] - peaks[use_peak]
        trace = np.full_like(self._selected_pixels, fill_value=np.nan,
                             dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, col in enumerate(self._selected_pixels):
                trace[i] = self._fit_gaussian(center, col)
                center = int(trace[i])
        fit = self._fit_polynomial(trace, self._selected_pixels, 2).best_fit
        return fit, dy, True

    def _check_for_completeness(self, vertical_positions):
        """
        Make sure there are no NaNs in the data to which you're fitting the
        Gaussian model.
        """
        nans = False
        for i, j in zip(self._selected_pixels, vertical_positions):
            count = len(
                np.where(np.isnan(self._master_trace[j-15:j+15, i]))[0])
            if count != 0:
                nans = True
        return nans

    def _find_next_trace(self, initial_trace: np.ndarray, dy) \
            -> (np.ndarray, np.ndarray, int, bool):
        """
        Find the next trace above a given starting trace. Returns an array of
        NaNs if the `_check_for_completeness` function returns False, meaning
        it skips traces if a part of them falls off of the detector.
        """
        starting_indices = (initial_trace + dy).astype(int)
        trace = np.full_like(self._selected_pixels, fill_value=np.nan,
                             dtype=float)
        nans = self._check_for_completeness(starting_indices)
        if not nans:
            starting_coordinates = zip(starting_indices, self._selected_pixels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, (center, col) in enumerate(starting_coordinates):
                    trace[i] = self._fit_gaussian(center, col)
            fit = self._fit_polynomial(
                trace, self._selected_pixels, 3).best_fit
            dy = trace[0] - initial_trace[0]
            if trace[-1] >= self._n_rows - dy:
                check = False
            else:
                check = True
            return fit, fit, dy, check
        else:
            return trace, starting_indices, dy*1.1, True

    def _find_valid_traces(self):
        """
        Move up and down along the traces, fitting each subsequent trace.
        """
        found_traces = []
        trace, dy, check = self._find_initial_trace()
        found_traces.append(trace)
        while check:
            store_trace, trace, dy, check = self._find_next_trace(trace, dy)
            found_traces.append(store_trace)
        return np.array(found_traces)

    def _fill_in_missing_traces(self):
        """
        Fit to the known trace positions, filling in traces which cross between
        detectors and a couple of traces above and below the detector edges.
        """
        found_traces = self._find_valid_traces()
        n_rows, n_cols = found_traces.shape
        vind = np.arange(n_rows)
        expanded_traces = np.zeros_like(found_traces)
        for i in range(n_cols):
            vslice = found_traces[:, i]
            fit = self._fit_polynomial(vslice, vind, 5)
            expanded_traces[:, i] = fit.eval(x=vind)
        return expanded_traces

    def _find_complete_traces(self):
        """
        Calculate the pixel positions of every complete trace, including those
        that go in between the detectors.
        """
        expanded_traces = self._fill_in_missing_traces()
        traces = np.zeros((expanded_traces.shape[0], self._n_cols))
        for i, trace in enumerate(expanded_traces):
            fit = self._fit_polynomial(trace, self._selected_pixels, 3)
            traces[i] = fit.eval(x=self._pixels)
        return traces

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad((0.5, 0.5, 0.5))
        axis.pcolormesh(self._master_trace, cmap=cmap)
        for trace in self._traces:
            axis.plot(self._pixels, trace, color='red', linewidth=0.5)
        plt.savefig(Path(file_path, 'order_traces.jpg'), dpi=600)
        plt.close(fig)

    @property
    def image(self) -> np.ndarray:
        return self._master_trace

    @property
    def header(self) -> dict:
        return self._header

    @property
    def traces(self) -> np.ndarray:
        return self._traces

    @property
    def pixels(self) -> np.ndarray:
        return self._pixels


class _OrderBounds:

    def __init__(self, master_trace: _MasterTrace, master_flat: _MasterFlat):
        self._master_trace = master_trace
        self._master_flat = master_flat
        self._upper_bounds, self._lower_bounds = self._calculate_order_bounds()
        self._slit_length = 2 * self._calculate_slit_half_length()
        self._n_pixels = len(self._master_trace.pixels)

    def _calculate_slit_half_length(self):
        slit_length = (self._master_trace.header['slit_length']
                       / self._master_trace.header['spatial_bin_scale'])
        return np.ceil(slit_length.value / 2).astype(int)

    def _make_artificial_flatfield(self) -> np.ndarray:
        artificial_flatfield = np.zeros_like(self._master_trace.image)
        slit_half_width = self._calculate_slit_half_length()
        for trace in self._master_trace.traces:
            for i in self._master_trace.pixels:
                j = np.round(trace[i]).astype(int)
                lower = j-slit_half_width
                if lower < 0:
                    lower = 0
                upper = j+slit_half_width
                if upper >= artificial_flatfield.shape[0]:
                    upper = artificial_flatfield.shape[0]
                artificial_flatfield[lower:upper, i] = 1
        return artificial_flatfield

    def _correlate_flat_with_artificial_flat(self):
        artificial_flatfield = self._make_artificial_flatfield()
        slit_length = self._calculate_slit_half_length()
        ccr = np.array(
            [np.nansum(self._master_flat.image
                       * np.roll(artificial_flatfield, i, axis=0))
             for i in range(-slit_length, slit_length)])
        offset = slit_length - ccr.argmax()
        return offset

    def _calculate_order_bounds(self):
        offset = self._correlate_flat_with_artificial_flat()
        half_width = self._calculate_slit_half_length()
        upper_bounds = np.round(self._master_trace.traces).astype(int)
        upper_bounds += half_width - offset
        lower_bounds = np.round(self._master_trace.traces).astype(int)
        lower_bounds -= half_width + offset
        return upper_bounds, lower_bounds

    def rectify_data(self, data: (_CCDData | _MasterBias | _MasterFlat |
                                  _MasterArc | _MasterTrace)) -> np.ndarray:
        extra = 2  # additional buffer around bounds
        rectified_data = []
        for ub, lb in zip(self._upper_bounds, self.lower_bounds):
            rectified_order = np.zeros(
                (self._slit_length+2*extra, self._n_pixels))
            for i in range(self._n_pixels):
                rectified_order[:, i] = data.image[lb[i]-extra:ub[i]+extra, i]
            rectified_data.append(rectified_order)
        return np.array(rectified_data)

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        cmap = plt.get_cmap('bone').copy()
        cmap.set_bad((0.5, 0.5, 0.5))
        axis.pcolormesh(self._master_flat.image, cmap=cmap, alpha=0.5)
        for ub, lb in zip(self._upper_bounds, self.lower_bounds):
            axis.fill_between(self._master_trace.pixels, ub, lb, color='red',
                              linewidth=0, alpha=0.25)
            axis.plot(self._master_trace.pixels, ub, color='red',
                      linewidth=0.5)
            axis.plot(self._master_trace.pixels, lb, color='red',
                      linewidth=0.5)
        plt.savefig(Path(file_path, 'order_edges.jpg'), dpi=600)
        plt.close(fig)

    @property
    def upper_bounds(self) -> np.ndarray:
        return self._upper_bounds

    @property
    def lower_bounds(self) -> np.ndarray:
        return self._lower_bounds

    @property
    def slit_length(self) -> int:
        return self._slit_length


class _WavelengthSolution:

    def __init__(self, master_arc: _MasterArc, master_flat: _MasterFlat,
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
        return np.ceil(slit_width.value / 2).astype(int)

    def _make_1d_spectra(self):
        """
        Make spatial averages of each spectrum to collapse them to one
        dimension. Also normalize them.
        """
        spectra = np.zeros(self._order_bounds.lower_bounds.shape, dtype=float)
        rectified_arcs = self._order_bounds.rectify_data(self._master_arc)
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
        echelle_angle = self._master_arc.header['echelle_angle'].value
        cross_disperser_angle = \
            self._master_arc.header['cross_disperser_angle'].value
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

        fit_centers = p(xc, yc)
        fit_edges = p(xe, ye)
        return fit_centers, fit_edges

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        cmap = plt.get_cmap('bone').copy()
        axis.pcolormesh(self._master_flat.image, cmap=cmap, alpha=0.5)
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
        plt.savefig(Path(file_path, 'order_numbers_and_wavelength_bounds.jpg'),
                    dpi=600)
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


class _ScienceData:

    def __init__(self, target: str, file_path: Path,
                 reduced_data_path: Path,
                 science_data_directory: str,
                 master_bias: _MasterBias, master_flat: _MasterFlat,
                 master_arc: _MasterArc, order_bounds: _OrderBounds,
                 wavelength_solution: _WavelengthSolution):
        self._target = target
        self._file_path = file_path
        self._reduced_data_path = reduced_data_path
        self._science_data_directory = science_data_directory
        self._master_bias = master_bias
        self._rectified_master_bias = order_bounds.rectify_data(master_bias)
        self._master_flat = master_flat
        self._rectified_master_flat = order_bounds.rectify_data(master_flat)
        self._master_arc = master_arc
        self._rectified_master_arc = order_bounds.rectify_data(master_arc)
        self._order_bounds = order_bounds
        self._wavelength_solution = wavelength_solution

    def reduce_science_data(self):
        raw_data = _retrieve_ccd_data_from_fits(
            self._file_path, remove_cosmic_rays=True)
        bias_subtracted_data = self._master_bias.subtract_bias(raw_data)
        flatfielded_data = self._master_flat.flat_correct(
            bias_subtracted_data)
        rectified_reduced_data = \
            self._order_bounds.rectify_data(flatfielded_data)
        self._make_fits(self._target, rectified_reduced_data, raw_data)

    # noinspection DuplicatedCode
    def _make_fits(self, target: str, reduced_data: np.ndarray,
                   raw_data: _CCDData):
        exposure_time = raw_data.header['exposure_time'].value
        primary = fits.PrimaryHDU(np.array(reduced_data) / exposure_time)
        header = primary.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
        header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
        header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
        header.append(('TARGET', f'{target}', 'name of target body'))
        header.append(('OBSDATE', f"{raw_data.header['datetime'].fits}",
                       'UTC datetime at start of observation'))
        header.append(('BUNIT', 'electrons/second',
                       'physical units of primary extension'))
        header.append(('EXPTIME', f"{raw_data.header['exposure_time'].value}",
                       'exposure time [seconds]'))
        header.append(('AIRMASS', f"{raw_data.header['airmass'].value}",
                       'airmass'))
        header.append(('OBSERVER', raw_data.header['observers'],
                       'last names of observers'))
        header.append(('SLITLEN', raw_data.header['slit_length'].value,
                       'slit length [arcsec]'))
        header.append(('SLITLENB', raw_data.header['slit_length_bins'].value,
                       'slit length [bins]'))
        header.append(('SLITWID', raw_data.header['slit_width'].value,
                       'slit width [arcsec]'))
        header.append(('SLITWIDB', raw_data.header['slit_width_bins'].value,
                       'slit width [bins]'))
        header.append(('XDNAME', raw_data.header['cross_disperser'],
                       'name of cross diserpser'))
        header.append(('XDANG', raw_data.header['cross_disperser_angle'].value,
                       'cross disperser angle [deg]'))
        header.append(('ECHANG', raw_data.header['echelle_angle'].value,
                       'echelle angle [deg]'))
        header.append(('SPABIN', raw_data.header['spatial_binning'].value,
                       'spatial binning [pix/bin]'))
        header.append(('SPEBIN', raw_data.header['spectral_binning'].value,
                       'spectral binning [pix/bin]'))
        header.append(('SPASCALE', raw_data.header['spatial_bin_scale'].value,
                       'spatial bin scale [arcsec/bin]'))
        header.append(('SPESCALE', raw_data.header['spectral_bin_scale'].value,
                       'spectral bin scale [arcsec/bin]'))
        header.append(('PIXWIDTH', raw_data.header['pixel_size'].value,
                       'pixel width [micron]'))

        rectified_raw_data = self._order_bounds.rectify_data(raw_data)
        raw_hdu = fits.ImageHDU(rectified_raw_data.squeeze(),
                                name='RAW')
        header = raw_hdu.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
        header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
        header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
        header.append(('BUNIT', 'electrons', 'extension physical units'))

        bias_hdu = fits.ImageHDU(self._rectified_master_bias.squeeze(),
                                 name='BIAS')
        header = bias_hdu.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
        header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
        header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
        header.append(('BUNIT', 'electrons', 'extension physical units'))

        flat_hdu = fits.ImageHDU(self._rectified_master_flat.squeeze(),
                                 name='FLAT')
        header = flat_hdu.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
        header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
        header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
        header.append(('BUNIT', 'none (normalized)',
                       'extension physical units'))

        arc_hdu = fits.ImageHDU(self._rectified_master_arc.squeeze(),
                                name='ARC')
        header = arc_hdu.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
        header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
        header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
        header.append(('BUNIT', 'none (normalized)',
                       'extension physical units'))

        echelle_orders_hdu = fits.ImageHDU(
            self._wavelength_solution.order_numbers.squeeze(),
            name='ECHELLE_ORDERS')
        wavelength_centers_hdu = fits.ImageHDU(
            self._wavelength_solution.centers.squeeze(),
            name='BIN_CENTER_WAVELENGTHS')
        header = wavelength_centers_hdu.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bin centers')
        header['NAXIS2'] = (header['NAXIS2'], 'number of echelle orders')
        header.append(('BUNIT', 'nm', 'wavelength physical unit'))
        wavelength_edges_hdu = fits.ImageHDU(
            self._wavelength_solution.edges.squeeze(),
            name='BIN_EDGE_WAVELENGTHS')
        header = wavelength_edges_hdu.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bin edges')
        header['NAXIS2'] = (header['NAXIS2'], 'number of echelle orders')
        header.append(('BUNIT', 'nm', 'wavelength physical unit'))
        hdul = fits.HDUList(
            [primary, raw_hdu, bias_hdu, flat_hdu, arc_hdu,
             echelle_orders_hdu, wavelength_centers_hdu, wavelength_edges_hdu])
        filename = raw_data.header['file_name'].replace('.fits',
                                                        '_reduced.fits')
        save_path = _make_directory(
            Path(self._reduced_data_path, self._science_data_directory))
        hdul.writeto(Path(save_path, filename), overwrite=True)
