from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

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


def flux_cmap():
    """
    Colormap "viridis" for displaying flux data.
    """
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad((0.75, 0.75, 0.75))
    return cmap