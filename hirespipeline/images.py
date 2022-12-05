from copy import deepcopy
from pathlib import Path

import astropy.units as u
import ccdproc
import numpy as np
from astropy.io import fits
from astropy.time import Time

from hirespipeline.general import airmass


def _determine_detector_layout(hdul: fits.HDUList) -> str:
    """
    If the HDUList has only one item, it's legacy data using the single
    detector, which I decided against supporting for now. If not, then it is
    the three-detector mosaic arrangement.
    """
    if len(hdul) == 1:
        raise Exception('This pipeline only works for the three-CCD detector '
                        'setup!')
    elif len(hdul) == 4:
        return 'mosaic'
    else:
        raise Exception('Unknown detector layout!')


def _reformat_observers(observer_names: str):
    """
    Reformat observer names so they are separated by commas. Cleans it up a
    bit.
    """
    return ', '.join(
        [name.strip() for name in observer_names.replace('/', ',').split(',')]
    )


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


def _get_header(file_path: Path) -> dict:
    """
    Retrieve ancillary metadata from the headers.
    """
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        binning = np.array(header['BINNING'].split(',')).astype(int)
        return {
            'file_name': file_path.name,
            'datetime': Time(header['DATE_BEG'], format='isot',
                             scale='utc'),
            'observers': _reformat_observers(header['OBSERVER']),
            'exposure_time': header['EXPTIME'] * u.second,
            'airmass': float(header['AIRMASS']) * airmass,
            'slit_length': header['SLITLEN'] * u.arcsec,
            'slit_length_bins': np.ceil(
                header['SLITLEN'] / header['SPATSCAL']) * u.bin,
            'slit_width': header['SLITWIDT'] * u.arcsec,
            'slit_width_bins': np.ceil(
                header['SLITWIDT'] / header['DISPSCAL']) * u.bin,
            'cross_disperser': header['XDISPERS'].lower(),
            'cross_disperser_angle':
                np.round(header['XDANGL'], 5) * u.degree,
            'echelle_angle': np.round(header['ECHANGL'], 5) * u.degree,
            'spatial_binning':
                (binning[0] * u.pixel / u.bin).astype(int),
            'spatial_bin_scale': header['SPATSCAL'] * u.arcsec / u.bin,
            'spectral_binning':
                (binning[1] * u.pixel / u.bin).astype(int),
            'spectral_bin_scale': header['DISPSCAL'] * u.arcsec / u.bin,
            'pixel_size': 15 * u.um,
            'processing_log': list(),
        }


def _remove_outliers(image: np.ndarray, percentile=99):
    """
    Remove outliers outside of a given range. Defaults to 1-99%.
    """
    image = deepcopy(image)
    low = np.nanpercentile(image, 100-percentile)
    high = np.nanpercentile(image, percentile)
    image[image < low] = low
    image[image > high] = high
    return image


def _get_data_and_uncertainty(
        file_path: Path,
        remove_cosmic_rays: bool = True) -> (u.Quantity, u.Quantity):
    """
    Retrieve the CCD image, gain correct and combine the three detectors
    into one, and calculate uncertainties for each pixel.
    """
    with fits.open(file_path) as hdul:
        data_image = np.full(_get_full_image_size(hdul),
                             fill_value=np.nan)
        uncertainty_image = np.full_like(data_image, fill_value=np.nan)
        header = hdul[0].header
        binning = np.array(header['BINNING'].split(',')).astype(int)
        for detector_number in range(1, 4):
            image_header = hdul[detector_number].header
            gain = header[f'CCDGN0{detector_number}'] * u.electron / u.adu
            read_noise = header[f'CCDRN0{detector_number}'] * u.electron
            detector_slice = _parse_mosaic_detector_slice(
                image_header['DATASEC'])
            detector_image = np.flipud(
                hdul[detector_number].data.T[detector_slice].astype(float)
            ) * u.adu
            if remove_cosmic_rays:
                detector_image, _ = ccdproc.cosmicray_lacosmic(
                    detector_image, gain=gain, readnoise=read_noise,
                    gain_apply=False)
                detector_image = _remove_outliers(detector_image)
                detector_image *= u.adu
            detector_image *= gain
            corner = _get_mosaic_detector_corner_coordinates(
                image_header, binning)
            n_rows, _ = detector_image.shape
            data_image[corner[0]:corner[0] + n_rows] = detector_image.value
            uncertainty_image[corner[0]:corner[0] + n_rows] = \
                np.sqrt(detector_image.value + read_noise.value ** 2)
    return data_image * u.electron, uncertainty_image * u.electron


class _CCDData:
    """
    Generic class to hold a CCD image, it's corresponding uncertainty, and a
    dictionary containing metadata "header" information.
    """
    def __init__(self, data: u.Quantity, uncertainty: u.Quantity or None,
                 header: dict or None):
        self._data = data
        self._uncertainty = uncertainty
        self._header = header

    @property
    def image(self) -> u.Quantity:
        return self._data

    @property
    def uncertainty(self) -> u.Quantity or None:
        return self._uncertainty

    @property
    def header(self) -> dict or None:
        return self._header


def _retrieve_ccd_data_from_fits(file_path: Path,
                                 remove_cosmic_rays: bool = False) -> _CCDData:
    """
    Convenience function to produce a _CCDImage object from a FITS filepath.
    """
    header = _get_header(file_path)
    if remove_cosmic_rays:
        header['processing_log'] += ['remove_cosmic_rays']
    data, uncertainty = _get_data_and_uncertainty(
        file_path, remove_cosmic_rays=remove_cosmic_rays)
    return _CCDData(data, uncertainty, header)
