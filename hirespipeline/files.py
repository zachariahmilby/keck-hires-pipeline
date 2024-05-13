import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from hirespipeline.graphics import flux_cmap, _parse_mosaic_detector_slice, \
    rcparams, calculate_norm


def check_if_directory_exists(directory: Path):
    if not directory.absolute().exists():
        raise OSError('Provided directory does not exist.')
    else:
        return directory.absolute()


def make_directory(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True)
    return directory


class _Files:
    """
    Object to hold FITS file paths and ensure that a directory exists.
    """
    def __init__(self, directory: Path, file_type: str = 'fits'):
        self._directory = check_if_directory_exists(directory)
        self._file_type = file_type

    @property
    def paths(self) -> list[Path]:
        return sorted(self._directory.glob(f'*.{self._file_type}*'))


class _FilesQuicklook:
    """
    This class makes a graphical quicklook product for each of the raw data
    files. It also makes a summary spreadsheet in CSV format for a brief
    overview of the data in each of the observation night directories.
    """

    def __init__(self, directory: str or Path):
        """
        Parameters
        ----------
        directory : str or Path
            Path to directory containing FITS files in '.fits.gz' format.
        """
        self._directory = check_if_directory_exists(Path(directory))

    @staticmethod
    def _make_info_text_block(hdul: fits.HDUList, filename: str):
        """
        Make strings of ancillary information for the graphic.
        """
        header = hdul[0].header
        datetime = Time(header['DATE_BEG'], format='isot', scale='utc').fits
        datetime = datetime.replace('T', ' ')
        left_column = fr'$\bf{{File\ Name\!:}}$ '
        left_column += fr'$\tt{{{filename}}}$' + '\n'
        left_column += fr'$\bf{{Observation\ Date\!:}}$ '
        left_column += fr'{datetime} UTC' + '\n'
        left_column += fr'$\bf{{Observers\!:}}$ {header["OBSERVER"]}' + '\n'
        left_column += fr'$\bf{{Observation\ Type\!:}}$ ' \
                       fr'{header["OBSTYPE"]}' + '\n'
        left_column += fr'$\bf{{Object\!:}}$ {header["OBJECT"]}' + '\n'
        left_column += fr'$\bf{{Target\!:}}$ {header["TARGNAME"]}' + '\n'
        left_column += fr'$\bf{{Exposure\ Time\!:}}$ ' \
                       fr'{header["EXPTIME"]:.2f} seconds' + '\n'
        left_column += fr'$\bf{{Airmass\!:}}$ ' \
                       fr'{float(header["AIRMASS"])}' + '\n'

        right_column = fr'$\bf{{Decker\!:}}$ {header["DECKNAME"]}' + '\n'
        right_column += fr'$\bf{{Lamp\!:}}$ {header["LAMPNAME"]}' + '\n'
        right_column += fr'$\bf{{Filter\ 1\!:}}$ {header["FIL1NAME"]}' + '\n'
        right_column += fr'$\bf{{Filter\ 2\!:}}$ {header["FIL2NAME"]}' + '\n'
        right_column += fr'$\bf{{Binning\!:}}$ {header["BINNING"]}' + '\n'
        right_column += fr'$\bf{{Cross\ Disperser:\!}}$ '
        right_column += fr'{header["XDISPERS"]}' + '\n'
        right_column += fr'$\bf{{Echelle\ Angle\!:}}$ '
        right_column += fr'${header["ECHANGL"]:.5f}\degree$' + '\n'
        right_column += fr'$\bf{{Cross\ Disperser\ Angle:\!}}$ '
        right_column += fr'${header["XDANGL"]:.4f}\degree$'

        return left_column, right_column

    def _save_quicklook(self, hdul: fits.HDUList, filename: str):
        """
        Save graphic quicklook to file.
        """
        cbar_formatter = ticker.ScalarFormatter()
        with plt.style.context(rcparams):
            left_column, right_column = self._make_info_text_block(hdul,
                                                                   filename)
            fig, axes = plt.subplots(
                4, 1, figsize=(6, 6),
                gridspec_kw={'height_ratios': [1, 2, 2, 2]},
                constrained_layout=True, clear=True)
            [axis.set_xticks([]) for axis in axes.ravel()]
            [axis.set_yticks([]) for axis in axes.ravel()]
            [axis.set_frame_on(False) for axis in axes.ravel()]
            for detector in range(1, 4):
                image_header = hdul[detector].header
                detector_slice = _parse_mosaic_detector_slice(
                    image_header['DATASEC'])
                image = np.flipud(
                        hdul[detector].data.T[detector_slice].astype(float))
                norm = calculate_norm(image)
                img = axes[4-detector].pcolormesh(image, cmap=flux_cmap(),
                                                  norm=norm, rasterized=True)
                cbar = plt.colorbar(img, ax=axes[4-detector], pad=0.02)
                cbar.ax.yaxis.set_major_formatter(cbar_formatter)
            axes[0].text(0, 1, left_column, ha='left', va='top',
                         transform=axes[0].transAxes, linespacing=1.5)
            axes[0].text(0.5, 1, right_column, ha='left', va='top',
                         transform=axes[0].transAxes, linespacing=1.5)
            fig.canvas.draw()

            if filename.split('.')[-1] == 'gz':
                savename = filename.replace('.fits.gz', '.jpg')
            else:
                savename = filename.replace('.fits', '.jpg')
            plt.savefig(Path(self._directory, savename))
            plt.close(fig)

    @staticmethod
    def _append_csv(df: pd.DataFrame, hdul: fits.HDUList, filename: str):
        header = hdul[0].header
        datetime = Time(header['DATE_BEG'], format='isot', scale='utc').fits
        data = {'Filename': filename,
                'Observation Date': datetime.replace('T', ' '),
                'Object': header['OBJECT'],
                'Target': header['TARGNAME'],
                'Exposure Time [s]': np.round(header['EXPTIME'], 2),
                'Observers': header['OBSERVER'],
                'Observation Type': header["OBSTYPE"],
                'Decker': header['DECKNAME'],
                'Lamp': header['LAMPNAME'],
                'Filter 1': header['FIL1NAME'],
                'Filter 2': header['FIL2NAME'],
                'Binning': header['BINNING'],
                'Airmass': header['AIRMASS'],
                'Echelle Angle [deg]': np.round(header['ECHANGL'], 5),
                'Cross Disperser Angle [deg]': np.round(header['XDANGL'], 4)
                }
        return df.append(data, ignore_index=True)

    def run(self, save_graphics: bool = True):
        files = sorted(self._directory.glob('*.fits'))
        files_zipped = sorted(self._directory.glob('*.fits.gz'))
        if (len(files) == 0) & (len(files_zipped) == 0):
            raise FileNotFoundError("No FITS files found in directory.")
        if len(files_zipped) != 0:
            files = files_zipped
        df = pd.DataFrame()
        for file in files:
            with fits.open(file) as hdul:
                if save_graphics:
                    self._save_quicklook(hdul, file.name)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=FutureWarning)
                    df = self._append_csv(df, hdul, file.name)
        savepath = Path(self._directory, 'file_information.csv')
        df.to_csv(savepath, index=False)


def create_quicklooks(directory: str or Path, save_graphics: bool = True):
    """
    Wrapper function to create quicklooks in a directory containing FITS files.

    Parameters
    ----------
    directory : str or Path
        The path to the directory.
    save_graphics : bool
        Whether or not to save graphics. If False, it will only save a summary
        CSV. Default is True.

    Returns
    -------
    None.
    """

    _FilesQuicklook(directory).run(save_graphics=save_graphics)
