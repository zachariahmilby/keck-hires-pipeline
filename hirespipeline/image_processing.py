import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import ccdproc
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time


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
        datetime = Time(header['DATE_BEG'], format='isot', scale='utc').fits
        return {
            'file_name': file_path.name,
            'datetime': datetime,
            'observers': header['OBSERVER'],
            'exposure_time': header['EXPTIME'],
            'airmass': float(header['AIRMASS']),
            'slit_length': header['SLITLEN'],
            'slit_length_bins': np.ceil(header['SLITLEN']/header['SPATSCAL']),
            'slit_width': header['SLITWIDT'],
            'slit_width_bins': np.ceil(header['SLITWIDT']/header['DISPSCAL']),
            'cross_disperser': header['XDISPERS'].lower(),
            'cross_disperser_angle': np.round(header['XDANGL'], 5),
            'echelle_angle': np.round(header['ECHANGL'], 5),
            'spatial_binning': int(binning[0]),
            'spatial_bin_scale': header['SPATSCAL'],
            'spectral_binning': int(binning[1]),
            'spectral_bin_scale': header['DISPSCAL'],
            'pixel_size': 15,
        }


def _combine_mosaic_image(file_path: Path) -> u.Quantity:
    """
    Combine mosaic image data into a single array with proper inter-detector
    spacing, and multiply each detector image by its corresponding gain so they
    are comparable.
    """
    with fits.open(file_path) as hdul:
        data_image = np.full(_get_full_image_size(hdul),
                             fill_value=np.nan)
        header = hdul[0].header
        binning = np.array(header['BINNING'].split(',')).astype(int)
        for detector_number in range(1, 4):
            image_header = hdul[detector_number].header
            gain = header[f'CCDGN0{detector_number}']
            detector_slice = _parse_mosaic_detector_slice(
                image_header['DATASEC'])
            detector_image = np.flipud(
                hdul[detector_number].data.T[detector_slice].astype(float))
            detector_image *= gain
            corner = _get_mosaic_detector_corner_coordinates(
                image_header, binning)
            n_rows, _ = detector_image.shape
            data_image[corner[0]:corner[0] + n_rows] = detector_image
    return data_image * u.electron


def _remove_header_info_for_masters(header: dict) -> dict:
    header = deepcopy(header)
    remove = ['datetime', 'exposure_time', 'airmass']
    [header.pop(key) for key in remove]
    return header


def _make_median_image(images: list[CCDData]) -> CCDData:
    """
    Make a median master image from a directory of FITS files.

    Parameters
    ----------
    images : list of CCDData objects
        A list containing images as CCDData objects.

    Returns
    -------
    median_image : CCDData
        A CCDData object of the median image.
    """
    combiner = ccdproc.Combiner(images)
    with warnings.catch_warnings():  # ignore all-NaN slices
        warnings.simplefilter('ignore', category=RuntimeWarning)
        median_image = combiner.median_combine()
    median_image.header = _remove_header_info_for_masters(images[0].header)
    return median_image
