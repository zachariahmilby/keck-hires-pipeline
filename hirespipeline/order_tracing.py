import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.utils.exceptions import AstropyUserWarning
from lmfit.models import GaussianModel, PolynomialModel
from lmfit.model import ModelResult
from scipy.ndimage import shift
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale

from hirespipeline.files import make_directory
from hirespipeline.general import shift_params, _log
from hirespipeline.graphics import (flux_cmap, flat_cmap, calculate_norm,
                                    nan_color, turn_off_axes)


class _OrderTraces:
    """
    Trace orders and find order boundaries.
    """
    def __init__(self,
                 master_trace: CCDData,
                 log_path: Path,
                 optimize: bool):
        """
        Parameters
        ----------
        master_trace : CCDData
            Master trace image.'
        log_path : Path
            Location of log file.
        optimize : bool
            Whether or not you want to optimize the traces.
        """
        self._master_trace = master_trace
        self._log_path = log_path
        self._normalized_data = self._process_data(master_trace.data)
        self._n_rows, self._n_cols = self._master_trace.data.shape
        self._selected_pixels = np.arange(0, self._n_cols, 128, dtype=int)
        self._starting_points = self._get_starting_points()
        self._pixels = np.arange(self._n_cols)
        if optimize:
            self._traces = self._refine_traces()
        else:
            self._traces = self._select_traces()

    @staticmethod
    def _process_data(data: np.ndarray) -> np.ndarray:
        """
        Normalize data to a range of 0 to 1.
        """
        ind = np.where(data == 0)
        data = minmax_scale(data)
        data[ind] = np.nan
        return data

    def _calculate_slit_half_length(self) -> int:
        """
        Calculate slit half-length in pixels.
        """
        slit_length = (self._master_trace.header['slit_length']
                       / self._master_trace.header['spatial_bin_scale'])
        return np.ceil(slit_length / 2).astype(int)

    def _fit_gaussian(self,
                      center: int,
                      col: int,
                      skip_if_nan: bool = False) -> float:
        """
        Fit a Gaussian function to a small vertical slice segment isolated from
        one of the traces. Returns just the center value.
        """
        center = int(center)
        y = self._normalized_data[center-15:center+16, col]
        x = np.arange(center-15, center+16)
        bad = np.where(np.isnan(y))[0]
        if len(bad) > 0:
            if skip_if_nan:
                return np.nan
        good = np.where(~np.isnan(y))[0]
        model = GaussianModel()
        params = model.guess(y[good], x=x[good])
        result = model.fit(y[good], params, x=x[good])
        center = result.params['center'].value
        return center

    @staticmethod
    def _fit_polynomial(y: np.ndarray,
                        x: np.ndarray,
                        degree: int) -> ModelResult:
        """
        Fit a polynomial model. Returns the full ModelResult.
        """
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
        vslice = self._normalized_data[:, self._selected_pixels[0]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            smoothed_slice = convolve(vslice, Gaussian1DKernel(stddev=1),
                                      boundary='extend')
        peaks, _ = find_peaks(smoothed_slice, height=0.2)
        peaks = peaks[1:-1]
        x = np.arange(len(peaks))

        gradient = np.gradient(peaks)
        bad, _ = find_peaks(gradient)

        if len(bad) > 0:
            bad_extended = np.sort(np.concatenate((bad - 1, bad, bad + 1)))
            x = np.arange(len(peaks))
            good = [i for i in x if i not in bad_extended]

            x = np.arange(len(good))
            start = bad_extended[0]
            peaks = peaks[good]

            fit = self._fit_polynomial(y=peaks[:start], x=x[:start], degree=5)
            pct = np.abs(1 - peaks / fit.eval(x=x))
            ind = np.where(pct > 0.05)[0]
            count = 0
            while len(ind) > 0:
                x[ind] += 1
                fit = self._fit_polynomial(y=peaks[:ind[0]], x=x[:ind[0]],
                                           degree=3)
                pct = np.abs(1 - peaks / fit.eval(x=x))
                ind = np.where(pct > 0.05)[0]
                count += 1
                if count > 100:
                    break

        fit = self._fit_polynomial(y=peaks, x=x, degree=5)
        extrapolated_peaks = fit.eval(x=np.arange(-10, len(peaks)+11))
        ind = np.where((extrapolated_peaks > 0) &
                       (extrapolated_peaks < self._n_rows))[0]

        return extrapolated_peaks[ind]

    def _find_initial_traces(self) -> np.ndarray:
        """
        Calculate initial traces using identified peaks in first column, which
        should be far enough for the whole trace to fall on the detector
        without any odd effects.
        """
        traces = np.full((self._starting_points.shape[0],
                          self._n_cols), fill_value=np.nan)
        for (i, starting_point) in enumerate(self._starting_points):
            trace = np.full(self._selected_pixels.shape[0], fill_value=np.nan)
            center = starting_point
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    for (j, col) in enumerate(self._selected_pixels):
                        trace[j] = self._fit_gaussian(int(center), col)  # noqa
                        center = trace[j]
                    fit = self._fit_polynomial(trace, self._selected_pixels, 3)
                    rsquare = 1 - fit.residual.var() / np.var(trace)
                    if rsquare > 0.999:
                        traces[i] = fit.eval(x=self._pixels)
                except (IndexError, TypeError, ValueError):
                    continue
        return traces

    def _fill_in_missing_traces(self) -> np.ndarray:
        """
        Fit to the known trace positions, filling in traces which cross between
        detectors and a couple of traces above and below the detector edges.
        """
        initial_traces = self._find_initial_traces()
        n_rows, n_cols = initial_traces.shape
        vind = np.arange(n_rows)
        vind_expanded = np.arange(-5, n_rows + 5 + 1, 1)
        expanded_traces = np.zeros(
            (vind_expanded.shape[0], initial_traces.shape[1]))
        for i in range(n_cols):
            vslice = initial_traces[:, i]
            fit = self._fit_polynomial(vslice, vind, 5)
            expanded_traces[:, i] = fit.eval(x=vind_expanded)
        return expanded_traces

    def _select_traces(self) -> np.ndarray:
        """
        Keep only those traces which are fully within the mosaic.
        """
        expanded_traces = self._fill_in_missing_traces()
        selected_traces = []
        for trace in expanded_traces:
            if (trace[0] > self._normalized_data.shape[0] - 1) | \
                    (trace[-1] < 0):
                continue
            else:
                selected_traces.append(trace)
        return np.array(selected_traces)

    def _refine_traces(self) -> np.ndarray:
        """
        Fit a Gaussian profile to each point in a trace, then fit a 5th-degree
        polynomial to the result to get the tightest best-fit trace.
        """
        _log(self._log_path, f'   Refining traces...')
        selected_traces = self._select_traces()
        refined_traces = np.zeros_like(selected_traces)
        slit_half_width = self._calculate_slit_half_length()
        shape = self._normalized_data.data.shape
        n = selected_traces.shape[0]
        for i, trace in enumerate(selected_traces):
            _log(self._log_path, f'      Order {i + 1}/{n}', new_line=False)
            ub = trace + slit_half_width  # noqa
            lb = trace - slit_half_width  # noqa
            if (np.max(ub) > shape[0] - 1) or (np.min(lb) < 0):
                refined_traces[i] = trace
            else:
                fit_centers = []
                for col in range(shape[1]):
                    center = self._fit_gaussian(
                        trace[col], col, skip_if_nan=True)
                    if np.isnan(center):
                        center = trace[col]
                    fit_centers.append(center)
                fit_centers = np.array(fit_centers)
                fit = self._fit_polynomial(
                    y=fit_centers, x=self._pixels, degree=5)
                refined_traces[i] = fit.eval(x=self._pixels)
        return refined_traces

    # noinspection DuplicatedCode
    def quality_assurance(self,
                          file_path: Path) -> None:
        """
        Generate a quality assurance graphic for ensuring traces were properly
        found.
        """
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True,
                                 clear=True)
        axis.set_facecolor(nan_color)
        turn_off_axes(axis)
        cmap = flux_cmap()
        display_data = self._normalized_data
        display_data[np.isnan(display_data)] = 0
        norm = calculate_norm(display_data)
        axis.pcolormesh(display_data, cmap=cmap, norm=norm, zorder=2)
        for trace in self._traces:
            axis.plot(self._pixels, trace, color='red', linewidth=0.5,
                      zorder=3)
        xlim = axis.get_xlim()
        axis.axvspan(xlim[0], xlim[-1], color=nan_color, zorder=1)
        axis.set_xlim(xlim)
        axis.set_ylim(np.nanmin(self._traces) - 10,
                      np.nanmax(self._traces) + 10)
        savepath = Path(file_path, 'quality_assurance', 'order_traces.jpg')
        make_directory(savepath.parent)
        plt.savefig(savepath, dpi=600)
        plt.close(fig)

    @property
    def traces(self) -> np.ndarray:
        """
        The trace vertical coordinates.
        """
        return self._traces

    @property
    def pixels(self) -> np.ndarray:
        """
        Pixel numbers corresponding to the trace vertical coordinates.
        """
        return self._pixels


class _OrderBounds:
    """
    Use the traces to find the bounds of each echelle order, including those
    that overlap.
    """
    def __init__(self,
                 order_traces: _OrderTraces,
                 master_flat: CCDData,
                 log_path: Path):
        """
        Parameters
        ----------
        order_traces: _OrderTraces
            Found order traces.
        master_flat: CCDData
            The master flatfield image.
        log_path : Path
            Location of log file.
        """
        self._order_traces = order_traces
        self._master_flat = master_flat
        self._log_path = log_path
        self._slit_length = 2 * self._calculate_slit_half_length()
        self._lower_bounds = self._calculate_order_lower_bound()
        self._n_pixels = len(self._order_traces.pixels)
        self._mask = np.ones_like(self._master_flat)

    def _calculate_slit_half_length(self) -> int:
        """
        Calculate slit half-length in pixels.
        """
        slit_length = (self._master_flat.header['slit_length']
                       / self._master_flat.header['spatial_bin_scale'])
        return np.ceil(slit_length / 2).astype(int)

    def _make_artificial_flatfield(self) -> np.ndarray:
        """
        Use the traces to construct an artifical binary flatfield.
        """
        artificial_flatfield = np.zeros_like(self._master_flat.data)
        slit_half_length = self._calculate_slit_half_length()
        for trace in self._order_traces.traces:
            for i in self._order_traces.pixels:
                j = np.round(trace[i]).astype(int)
                lower = j-slit_half_length
                if lower < 0:
                    lower = 0
                upper = j+slit_half_length+1
                if upper >= artificial_flatfield.shape[0]:
                    upper = artificial_flatfield.shape[0]
                artificial_flatfield[lower:upper, i] = 1
        return artificial_flatfield

    def _correlate_flat_with_artificial_flat(self) -> np.ndarray:
        """
        Cross-correlate the binary flatfield with the observed master flat to
        determine any offset between the location of the trace and the edges of
        the flat.
        """
        artificial_flatfield = self._make_artificial_flatfield()
        half_slit = self._calculate_slit_half_length()
        shifts = np.linspace(-half_slit, half_slit, 100)
        ccr = np.array(
            [np.nansum(self._master_flat.data
                       * shift(artificial_flatfield, [i, 0], **shift_params))
             for i in shifts])
        return shifts[ccr.argmax()]

    def _calculate_order_lower_bound(self) -> np.ndarray:
        """
        Calcualte the lower bounds of each order.
        """
        offset = self._correlate_flat_with_artificial_flat()
        half_width = self._calculate_slit_half_length()
        lower_bounds = self._order_traces.traces - half_width + offset
        return lower_bounds

    def _remove_overlap(self,
                        data : np.ndarray,
                        uncertainty: np.ndarray,
                        n: int or float) -> tuple[np.ndarray, np.ndarray]:
        """
        Find areas on the detector where order bounds overlap and set them to
        zero.
        """
        shape = data.shape
        lb = self.lower_bounds + n
        ub = lb + self._slit_length
        for lbi, ubi in zip(lb[1:], ub[:-1]):
            if lbi[0] < ubi[0]:
                for col in range(shape[1]):
                    s_ = np.s_[np.floor(lbi[col]).astype(int)-1:
                               np.ceil(ubi[col]).astype(int)+2, col]
                    data[s_] = np.nan
                    uncertainty[s_] = np.nan
        return data, uncertainty

    def _get_padded_data(self,
                         ccd_data: CCDData,
                         n: int = 100) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Add some extra rows of zeros so that the full slit width can be
        captured near the detector edges.
        """
        data = ccd_data.data
        unc = ccd_data.uncertainty.array
        extra = np.zeros((n, data.shape[1]))
        data = np.concatenate((extra, data, extra), axis=0)
        unc = np.concatenate((extra, unc, extra), axis=0)
        data, unc = self._remove_overlap(data, unc, n)
        return data, unc, n

    def rectify(self,
                ccd_data: CCDData) -> CCDData:
        """
        Rectify a CCDData object using the calculated order bounds.
        """
        rectified_data = []
        rectified_uncertainty = []
        data, uncertainty, n = self._get_padded_data(ccd_data=ccd_data)
        nans = np.isnan(data)
        data[nans] = 0.0
        uncertainty[nans] = 0.0
        mask = data.copy()
        mask[np.where(mask > 0)] = 1.0
        mask[np.where(mask < 0)] = 0.0
        self._mask = mask
        s_ = np.s_[:self._slit_length + 1]
        for check, lb in enumerate(self.lower_bounds + n):
            rectified_order_data = np.zeros(
                (self._slit_length + 1, self._n_pixels))
            rectified_order_unc = np.zeros(
                (self._slit_length + 1, self._n_pixels))
            for i in range(self._n_pixels):
                shifted_data = shift(data[:, i], -lb[i], **shift_params)
                shifted_unc = shift(uncertainty[:, i], -lb[i], **shift_params)
                rectified_order_data[:, i] = shifted_data[s_]
                rectified_order_unc[:, i] = shifted_unc[s_]
            rectified_data.append(rectified_order_data)
            rectified_uncertainty.append(rectified_order_unc)
        rectified_data = np.array(rectified_data)
        rectified_uncertainty = StdDevUncertainty(
            np.array(rectified_uncertainty))
        return CCDData(data=rectified_data,
                       uncertainty=rectified_uncertainty,
                       unit=ccd_data.unit,
                       header=ccd_data.header.copy())

    # noinspection DuplicatedCode
    def quality_assurance(self,
                          file_path: Path) -> None:
        """
        Generate a quality assurance graphic for ensuring order bounds were
        properly found.
        """
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True,
                                 clear=True)
        axis.set_facecolor(nan_color)
        turn_off_axes(axis)
        cmap = flat_cmap()
        norm = calculate_norm(self._master_flat.data)
        axis.pcolormesh(self._master_flat.data * self._mask, cmap=cmap,
                        norm=norm, alpha=0.5, zorder=2)
        for lb in self.lower_bounds:
            ub = lb + self._slit_length
            axis.fill_between(self._order_traces.pixels, ub, lb, color='red',
                              linewidth=0, alpha=0.25, zorder=3)
            axis.plot(self._order_traces.pixels, ub, color='red',
                      linewidth=0.5, zorder=4)
            axis.plot(self._order_traces.pixels, lb, color='red',
                      linewidth=0.5, zorder=4)
        xlim = axis.get_xlim()
        axis.axvspan(xlim[0], xlim[-1], color=nan_color, zorder=1)
        axis.set_xlim(xlim)
        axis.set_ylim(np.min(self.lower_bounds) - 10,
                      np.max(self.lower_bounds) + self._slit_length + 10)
        savepath = Path(file_path, 'quality_assurance', 'order_edges.jpg')
        make_directory(savepath.parent)
        plt.savefig(savepath, dpi=600)
        plt.close(fig)

    @property
    def lower_bounds(self) -> np.ndarray:
        """
        The lower bounds of the echelle orders.
        """
        return self._lower_bounds

    @property
    def slit_length(self) -> int:
        """
        The length (vertial size) of the slit in pixels.
        """
        return self._slit_length

    @property
    def overlap_mask(self) -> np.ndarray:
        """
        The binary mask which will set any pixels contained in two or more
        orders to zero.
        """
        return self._mask
