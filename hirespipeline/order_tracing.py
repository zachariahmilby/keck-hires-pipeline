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
        self._starting_points = self._get_starting_points()
        self._pixels = np.arange(self._n_cols)
        self._traces = self._select_traces()

    def _fit_gaussian(self, center, col):
        center = int(center)
        y = self._master_trace[center-15:center+16, col]
        x = np.arange(center-15, center+16)
        good = np.where(~np.isnan(y))[0]
        model = GaussianModel()
        params = model.guess(y[good], x=x[good])
        result = model.fit(y[good], params, x=x[good])
        center = result.params['center'].value
        return center

    @staticmethod
    def _fit_polynomial(y: np.ndarray, x: np.ndarray, degree: int):
        model = PolynomialModel(degree=degree)
        ind = np.isfinite(y)
        params = model.guess(y[ind], x=x[ind])
        result = model.fit(y[ind], params, x=x[ind])
        return result

    def _get_starting_points(self) -> np.ndarray:
        """
        Find peaks from trace image, fit a polynomial to their separation, then
        extrapolate for the full vertical pixel range of the composite image.
        """
        vslice = self._master_trace[:, 0]
        peaks, _ = find_peaks(vslice, height=0.2)
        index = np.arange(len(peaks))
        fit = self._fit_polynomial(peaks, index, 3)
        extrapolated_peaks = fit.eval(x=np.arange(-10, len(peaks)+11))
        ind = np.where((extrapolated_peaks > 0) &
                       (extrapolated_peaks < self._n_rows))[0]
        return extrapolated_peaks[ind]

    def _find_initial_traces(self) -> (np.ndarray, int, bool):
        """
        Calculate an initial trace using the third identified peak (should be
        far enough for the whole trace to fall on the detector without any odd
        effects. Also returns the initial spacing between this trace and its
        adjacent traces.
        """
        traces = np.full((self._starting_points.shape[0],
                          self._n_cols), fill_value=np.nan)
        for i, starting_point in enumerate(self._starting_points):
            trace = np.full(self._selected_pixels.shape[0], fill_value=np.nan)
            center = starting_point
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    for j, col in enumerate(self._selected_pixels):
                        trace[j] = self._fit_gaussian(center, col)
                        center = trace[j]
                    fit = self._fit_polynomial(trace, self._selected_pixels, 3)
                    rsquare = 1 - fit.residual.var() / np.var(trace)
                    if rsquare > 0.9999:
                        traces[i] = fit.eval(x=self._pixels)
                except (IndexError, TypeError, ValueError):
                    continue
        return traces

    def _fill_in_missing_traces(self):
        """
        Fit to the known trace positions, filling in traces which cross between
        detectors and a couple of traces above and below the detector edges.
        """
        initial_traces = self._find_initial_traces()
        n_rows, n_cols = initial_traces.shape
        vind = np.arange(n_rows)
        expanded_traces = np.zeros_like(initial_traces)
        for i in range(n_cols):
            vslice = initial_traces[:, i]
            fit = self._fit_polynomial(vslice, vind, 5)
            expanded_traces[:, i] = fit.eval(x=vind)
        return expanded_traces

    def _select_traces(self) -> np.ndarray:
        """
        Keep only those traces which are fully within the mosaic.
        """
        expanded_traces = self._fill_in_missing_traces()
        selected_traces = []
        for trace in expanded_traces:
            if (trace[0] > 0) & (trace[-1] < self._master_trace.shape[0] - 1):
                selected_traces.append(trace)
        return np.array(selected_traces)

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

    @staticmethod
    def _get_padded_data(ccd_data: CCDData,
                         n: int = 100) -> (np.ndarray, np.ndarray):
        """
        Add some extra rows of NaNs so that the full slit width can be captured
        near the detector edges.
        """
        data = ccd_data.data
        unc = ccd_data.uncertainty.array
        extra = np.full((n, data.shape[1]), fill_value=np.nan)
        data = np.concatenate((extra, data, extra), axis=0)
        unc = np.concatenate((extra, unc, extra), axis=0)
        return data, unc, n

    def rectify(self, ccd_data: CCDData) -> CCDData:
        """
        Rectify a CCDData object using the calculated order bounds.
        """
        rectified_data = []
        rectified_uncertainty = []
        data, uncertainty, n = self._get_padded_data(ccd_data=ccd_data)
        for ub, lb in zip(self._upper_bounds + n, self.lower_bounds + n):
            rectified_order_data = np.zeros(
                (self._slit_length + 1, self._n_pixels))
            rectified_order_unc = np.zeros(
                (self._slit_length + 1, self._n_pixels))
            for i in range(self._n_pixels):
                rectified_order_data[:, i] = data[lb[i]:ub[i], i]
                rectified_order_unc[:, i] = uncertainty[lb[i]:ub[i], i]
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
