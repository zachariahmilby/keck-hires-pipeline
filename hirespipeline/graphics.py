from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from hirespipeline.general import package_directory

"""Location of Matplotlib runtime configuration."""
rcparams = Path(package_directory, 'anc', 'rcparams.mplstyle')

nan_color = (0.75, 0.75, 0.75)


def turn_off_axes(axis: plt.Axes) -> None:
    """
    Turn off ticks and tick numbers.
    """
    axis.set_frame_on(False)
    axis.set_xticks([])
    axis.set_yticks([])


def calculate_norm(data: np.ndarray,
                   percentile: int | float = 99.0) -> colors.Normalize:
    """
    Calculate a nth-percentile linear normalization.
    """
    tempdata = data.copy().flatten()
    tempdata = tempdata[np.where(tempdata > 0)]
    vmin = np.nanpercentile(tempdata, 100-percentile).astype(float)
    vmax = np.nanpercentile(tempdata, percentile).astype(float)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _get_cmap(name: str) -> colors.Colormap:
    """
    Get a Matplotlib colormap and set the nan color.
    """
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(nan_color)
    return cmap


def bias_cmap() -> colors.Colormap:
    """
    Colormap "cividis" for displaying bias data.
    """
    return _get_cmap('cividis')


def arc_cmap() -> colors.Colormap:
    """
    Colormap "inferno" for displaying bias data.
    """
    return _get_cmap('inferno')


def flux_cmap() -> colors.Colormap:
    """
    Colormap "viridis" for displaying flux data.
    """
    return _get_cmap('viridis')


def flat_cmap() -> colors.Colormap:
    """
    Colormap "bone" for displaying flatfield data.
    """
    return _get_cmap('bone')


def _parse_legacy_detector_slice(header: fits.Header) -> np.s_:
    """"
    Extract the Python slice which trims detector edges in the spatial
    dimension for legacy data.
    """
    slice0 = header['PREPIX']
    slice1 = header['NAXIS1'] - header['POSTPIX']
    return np.s_[:, slice0:slice1]


def _parse_mosaic_detector_slice(slice_string: str) -> np.s_:
    """
    Extract the Python slice which trims detector edges in the spatial
    dimension for mosaic data.
    """
    indices = np.array(slice_string.replace(':', ',').replace('[', '')
                       .replace(']', '').split(',')).astype(int)
    indices[[0, 2]] -= 1
    return np.s_[indices[0]:indices[1], indices[2]:indices[3]]


def _get_mosaic_detector_corner_coordinates(
        image_header: fits.header.Header,
        binning: np.ndarray) -> np.ndarray:
    """
    Determine the relative physical coordinate of a mosaic detector's lower
    left corner. These coordinates let you replicate the actual physical
    layout of the detectors including the physical separation between them.
    """
    n_rows, n_columns = image_header['CRVAL1G'], image_header['CRVAL2G']
    spatial_coordinate = \
        np.abs(np.ceil(2048 - n_rows - 1) / binning[0]).astype(int)
    spectral_coordinate = np.ceil(n_columns - 1).astype(int)
    return np.array([spatial_coordinate, spectral_coordinate])


def _get_full_image_size(hdul: fits.HDUList) -> (int, int):
    """
    Calculate the dimensions of a full image containing all three detectors
    with proper inter-detector spacing.
    """
    binning = np.array(hdul[0].header['BINNING'].split(',')).astype(int)
    detector_slice = _parse_mosaic_detector_slice(
        hdul[1].header['DATASEC'])
    detector_image = np.flipud(hdul[1].data.T[detector_slice])
    n_rows, n_cols = detector_image.shape
    top_corner = _get_mosaic_detector_corner_coordinates(
                        hdul[3].header, binning)[0]
    return top_corner + n_rows, n_cols
