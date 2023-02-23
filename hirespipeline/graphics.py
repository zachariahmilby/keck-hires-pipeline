from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from hirespipeline.general import package_directory

"""Location of Matplotlib runtime configuration."""
rcparams = Path(package_directory, 'anc', 'rcparams.mplstyle')


def turn_off_axes(axis: plt.Axes):
    """
    Turn off ticks and tick numbers.
    """
    axis.set_xticks([])
    axis.set_yticks([])


def calculate_norm(data: np.ndarray, percentile=99):
    """
    Calculate a 99th-percentile linear normalization.
    """
    vmin = np.nanpercentile(data[:, :3584], 100-percentile)
    vmax = np.nanpercentile(data[:, :3584], percentile)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def bias_cmap():
    """
    Colormap "cividis" for displaying bias data.
    """
    cmap = plt.get_cmap('cividis').copy()
    cmap.set_bad((0.75, 0.75, 0.75))
    return cmap


def arc_cmap():
    """
    Colormap "inferno" for displaying bias data.
    """
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad((0.75, 0.75, 0.75))
    return cmap


def flux_cmap():
    """
    Colormap "viridis" for displaying flux data.
    """
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad((0.75, 0.75, 0.75))
    return cmap


def _parse_mosaic_detector_slice(slice_string: str) -> tuple[slice, slice]:
    """
    Extract the Python slice which trims detector edges in the spatial
    dimension for mosaic data.
    """
    indices = np.array(slice_string.replace(':', ',').replace('[', '')
                       .replace(']', '').split(',')).astype(int)
    indices[[0, 2]] -= 1
    return slice(indices[0], indices[1], 1), slice(indices[2], indices[3],
                                                   1)


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
