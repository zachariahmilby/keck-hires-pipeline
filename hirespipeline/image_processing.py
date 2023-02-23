import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import ccdproc
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.time import Time

from hirespipeline.airmass_extinction import _extinction_correct
from hirespipeline.general import readnoise
from hirespipeline.graphics import _parse_mosaic_detector_slice, \
    _get_full_image_size, _get_mosaic_detector_corner_coordinates
from hirespipeline.order_tracing import _OrderBounds
from hirespipeline.wavelength_solution import _WavelengthSolution


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


def _get_images_from_directory(
        directory: Path, remove_cosmic_rays: bool = False) -> list[CCDData]:
    """
    Make a list of CCDData objects of combined mosaic data.
    """
    files = sorted(directory.glob('*.fits*'))
    images = []
    for file in files:
        header = _get_header(file)
        data = CCDData(_combine_mosaic_image(file), header=header)
        if remove_cosmic_rays:
            data = ccdproc.cosmicray_lacosmic(data)
        data.data[np.where(data.data == 0)] = np.nan  # saturated pixels to NaN
        data_with_uncertainty = ccdproc.create_deviation(
            data, readnoise=readnoise, disregard_nan=True)
        images.append(data_with_uncertainty)
    return images


def _remove_header_info_for_masters(header: dict) -> dict:
    """
    Remove datetime, exposure_time and airmass from the header for median
    images, since they are no longer meaningful.
    """
    header = deepcopy(header)
    remove = ['datetime', 'exposure_time', 'airmass']
    [header.pop(key) for key in remove]
    return header


def _calculate_median_uncertainty(
        uncertainty: np.ndarray) -> StdDevUncertainty:
    """
    Calcualte uncertainty of median along an axis using the formula detailed in
    https://mathworld.wolfram.com/StatisticalMedian.html
    """
    n = uncertainty.shape[0]
    mean_unc = np.sqrt(np.sum(uncertainty**2, axis=0)) / n
    median_unc = mean_unc * np.sqrt(np.pi * (2 * n + 1) / (4 * n))
    return StdDevUncertainty(median_unc)


def _make_median_bias(bias_images: list[CCDData]) -> CCDData:
    """
    Make a median bias image from a list of CCDData objects.
    """
    warnings.simplefilter('ignore', category=RuntimeWarning)
    median_ccd = ccdproc.combine(bias_images, method='median')
    median_ccd.header = _remove_header_info_for_masters(
        bias_images[0].header.copy())
    return median_ccd


def _make_median_flux(flux_images: list[CCDData]) -> CCDData:
    """
    Make a median flux image from a list of CCDData objects. This is
    appropriate for flats or arcs, but not for bias.
    """
    with warnings.catch_warnings():  # ignore all-NaN slices
        warnings.simplefilter('ignore', category=RuntimeWarning)
        data = np.array([ccd.data for ccd in flux_images])
        # scale each image by its exposure time before calculating median
        scales = np.array([1./ccd.header['exposure_time']
                           for ccd in flux_images])
        scales = np.tile(scales[:, None, None],
                         (1, data.shape[1], data.shape[2]))
        median_data = np.nanmedian(data*scales, axis=0)
        uncertainty = np.array([ccd.uncertainty.array for ccd in flux_images])
        median_uncertainty = _calculate_median_uncertainty(
            uncertainty*scales)
        median_ccd = CCDData(data=median_data, uncertainty=median_uncertainty,
                             unit='electron')
        median_ccd.header = _remove_header_info_for_masters(
            flux_images[0].header.copy())
    return median_ccd


def _make_master_bias(file_directory: Path) -> CCDData:
    """
    Wrapper function to make a master bias detector image.
    """
    bias_images = _get_images_from_directory(Path(file_directory, 'bias'))
    master_bias = _make_median_bias(bias_images)
    return master_bias


def _make_master_flux(file_directory: Path, flux_type: str,
                      master_bias: CCDData) -> CCDData:
    """
    Wrapper function to make a master flat or arc detector image.
    """
    flux_images = _get_images_from_directory(Path(file_directory, flux_type))
    flux_images = [ccdproc.subtract_bias(image, master=master_bias)
                   for image in flux_images]
    master_flux = _make_median_flux(flux_images)
    return master_flux


def _make_master_trace(file_directory: Path, master_bias: CCDData) -> CCDData:
    """
    Wrapper function to load the first trace file and make a master order trace
    image.
    """
    trace_image = _get_images_from_directory(Path(file_directory, 'trace'))[0]
    master_trace = ccdproc.subtract_bias(trace_image, master=master_bias)
    return master_trace


def _process_science_data(
        file_directory: Path, sub_directory: str,
        master_bias: CCDData, master_flat: CCDData,
        order_bounds: _OrderBounds,
        wavelength_solution: _WavelengthSolution) -> ([CCDData], [str]):
    """
    Wrapper function to apply all of the reduction steps to science data in a
    supplied directory.
    """
    print(f'      Loading data and removing cosmic rays...')
    science_images = _get_images_from_directory(
        Path(file_directory, sub_directory), remove_cosmic_rays=True)
    count = len(science_images)
    reduced_science_images = []
    filenames = []
    for i, ccd_image in enumerate(science_images):
        filename = ccd_image.header['file_name']
        print(f'      Reducing image {i + 1}/{count}: {filename}')
        rectified_data = order_bounds.rectify(ccd_data=ccd_image)
        bias_subtracted_data = ccdproc.subtract_bias(
            rectified_data, master=master_bias)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            flat_corrected_data = ccdproc.flat_correct(
                bias_subtracted_data, flat=master_flat,
                norm_value=np.nanmean(master_flat.data))
            flux_data = flat_corrected_data.divide(
                flat_corrected_data.header['exposure_time'] * u.second)
            flux_data.header = flat_corrected_data.header.copy()
            extinction_corrected_data = _extinction_correct(
                rectified_data=flux_data,
                wavelength_solution=wavelength_solution)
            reduced_science_images.append(extinction_corrected_data)
            filenames.append(filename)
    return reduced_science_images, filenames
