import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import CCDData, StdDevUncertainty
from lmfit.models import GaussianModel, PolynomialModel
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale

from hirespipeline.files import make_directory


class _OrderTraces:
    """
    Trace orders and find order boundaries.
    """

    def __init__(self, master_trace: CCDData):
        self._master_trace = minmax_scale(master_trace.data)
        self._n_rows, self._n_cols = self._master_trace.data.shape
        self._selected_pixels = np.arange(0, self._n_cols, 128, dtype=int)
        self._pixels = np.arange(self._n_cols)
        self._traces = self._find_complete_traces()

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
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True,
                                 clear=True)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad((0.5, 0.5, 0.5))
        axis.pcolormesh(self._master_trace, cmap=cmap)
        for trace in self._traces:
            axis.plot(self._pixels, trace, color='red', linewidth=0.5)
        savepath = Path(file_path, 'quality_assurance', 'order_traces.jpg')
        make_directory(savepath.parent)
        plt.savefig(savepath, dpi=600)
        plt.close(fig)

    @property
    def traces(self) -> np.ndarray:
        return self._traces

    @property
    def pixels(self) -> np.ndarray:
        return self._pixels


class _OrderBounds:

    def __init__(self, order_traces: _OrderTraces, master_flat: CCDData):
        self._order_traces = order_traces
        self._master_flat = master_flat
        self._upper_bounds, self._lower_bounds = self._calculate_order_bounds()
        self._slit_length = 2 * self._calculate_slit_half_length()
        self._n_pixels = len(self._order_traces.pixels)

    def _calculate_slit_half_length(self):
        slit_length = (self._master_flat.header['slit_length']
                       / self._master_flat.header['spatial_bin_scale'])
        return np.ceil(slit_length / 2).astype(int)

    def _make_artificial_flatfield(self) -> np.ndarray:
        artificial_flatfield = np.zeros_like(self._master_flat.data)
        slit_half_width = self._calculate_slit_half_length()
        for trace in self._order_traces.traces:
            for i in self._order_traces.pixels:
                j = np.round(trace[i]).astype(int)
                lower = j-slit_half_width
                if lower < 0:
                    lower = 0
                upper = j+slit_half_width+1
                if upper >= artificial_flatfield.shape[0]:
                    upper = artificial_flatfield.shape[0]
                artificial_flatfield[lower:upper, i] = 1
        return artificial_flatfield

    def _correlate_flat_with_artificial_flat(self):
        artificial_flatfield = self._make_artificial_flatfield()
        slit_length = self._calculate_slit_half_length()
        ccr = np.array(
            [np.nansum(self._master_flat.data
                       * np.roll(artificial_flatfield, i, axis=0))
             for i in range(-slit_length, slit_length)])
        offset = slit_length - ccr.argmax()
        return offset

    def _calculate_order_bounds(self):
        offset = self._correlate_flat_with_artificial_flat()
        half_width = self._calculate_slit_half_length()
        upper_bounds = np.round(self._order_traces.traces).astype(int)
        upper_bounds += half_width - offset + 1
        lower_bounds = np.round(self._order_traces.traces).astype(int)
        lower_bounds -= half_width + offset
        return upper_bounds, lower_bounds

    def rectify(self, ccd_data: CCDData) -> CCDData:
        """
        Rectify a CCDData object using the calculated order bounds.
        """
        rectified_data = []
        rectified_uncertainty = []
        for ub, lb in zip(self._upper_bounds, self.lower_bounds):
            rectified_order_data = np.zeros(
                (self._slit_length + 1, self._n_pixels))
            rectified_order_unc = np.zeros(
                (self._slit_length + 1, self._n_pixels))
            for i in range(self._n_pixels):
                rectified_order_data[:, i] = ccd_data.data[lb[i]:ub[i], i]
                rectified_order_unc[:, i] = \
                    ccd_data.uncertainty.array[lb[i]:ub[i], i]
            rectified_data.append(rectified_order_data)
            rectified_uncertainty.append(rectified_order_unc)
        rectified_data = np.array(rectified_data)
        rectified_uncertainty = StdDevUncertainty(
            np.array(rectified_uncertainty))
        return CCDData(data=rectified_data, uncertainty=rectified_uncertainty,
                       unit=ccd_data.unit, header=ccd_data.header.copy())

    # noinspection DuplicatedCode
    def quality_assurance(self, file_path: Path):
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True,
                                 clear=True)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        cmap = plt.get_cmap('bone').copy()
        cmap.set_bad((0.5, 0.5, 0.5))
        axis.pcolormesh(self._master_flat.data, cmap=cmap, alpha=0.5)
        for ub, lb in zip(self._upper_bounds, self.lower_bounds):
            axis.fill_between(self._order_traces.pixels, ub, lb, color='red',
                              linewidth=0, alpha=0.25)
            axis.plot(self._order_traces.pixels, ub, color='red',
                      linewidth=0.5)
            axis.plot(self._order_traces.pixels, lb, color='red',
                      linewidth=0.5)
        savepath = Path(file_path, 'quality_assurance', 'order_edges.jpg')
        make_directory(savepath.parent)
        plt.savefig(savepath, dpi=600)
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
