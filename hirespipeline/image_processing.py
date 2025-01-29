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
from hirespipeline.general import (readnoise, low_gain, high_gain,
                                   detector_vertical_gaps, bad_columns, _log)
from hirespipeline.graphics import _parse_mosaic_detector_slice
from hirespipeline.order_tracing import _OrderBounds
from hirespipeline.wavelength_solution import _WavelengthSolution


def _parse_cross_disperser(cross_disperser: str) -> str:
    """
    Determine name of cross disperser for FITS header. Essentially just
    converts from 'uv' to 'blue' for HIRESb.
    """
    if cross_disperser.lower() == 'red':
        return 'red'
    elif cross_disperser.lower() == 'uv':
        return 'blue'


def determine_detector_layout(hdul: fits.HDUList) -> str:
    """
    If the HDUList has only one item, it's legacy data using the single
    detector. If not, then it is the three-detector mosaic arrangement.
    """
    if len(hdul) == 1:
        return 'legacy'
    elif len(hdul) == 4:
        return 'mosaic'
    else:
        raise Exception('Unknown detector layout!')


def _get_header(file_path: Path,
                slit_length: float | None,
                slit_width: float | None,
                spatial_binning: float | None,
                spectral_binning: float | None) -> dict:
    """
    Retrieve ancillary metadata from the headers.
    """
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        binning = np.array(header['BINNING'].split(',')).astype(int)
        date_fmt = dict(format='isot', scale='utc')
        if len(hdul) > 1:
            datetime = Time(header['DATE_BEG'], **date_fmt).fits
            detector_layout = 'mosaic'
            pixel_size = 15
        else:
            datetime = Time(header['DATE'], **date_fmt)
            detector_layout = 'legacy'
            pixel_size = 24
        try:
            if slit_length is None:
                slit_length = header['SLITLEN']
        except KeyError:
            raise Exception("If using files with non-unique filenames, e.g., "
                            "hires00001.fits, you must specify the slit "
                            "length (in arcseconds) when instantiating the "
                            "HIRESPipeline class.")
        try:
            if slit_width is None:
                slit_width = header['SLITWIDT']
        except KeyError:
            raise Exception("If using files with non-unique filenames, e.g., "
                            "hires00001.fits, you must specify the slit "
                            "width (in arcseconds) when instantiating the "
                            "HIRESPipeline class.")
        try:
            if spatial_binning is None:
                spatial_scale = header['SPATSCAL']
            else:
                spatial_scale = 0.119 * spatial_binning
        except KeyError:
            raise Exception("If using files with non-unique filenames, e.g., "
                            "hires00001.fits, you must specify spatial "
                            "binning when instantiating the HIRESPipeline "
                            "class.")
        try:
            if spectral_binning is None:
                spectral_scale = header['DISPSCAL']
            else:
                spectral_scale = 0.179 * spectral_binning
        except KeyError:
            raise Exception("If using files with non-unique filenames, e.g., "
                            "hires00001.fits, you must specify spectral "
                            "binning when instantiating the HIRESPipeline "
                            "class.")

        try:
            skypa = header['SKYPA']
        except KeyError:
            skypa = 'unknown'

        return {
            'file_name': file_path.name,
            'datetime': datetime,
            'observers': header['OBSERVER'],
            'exposure_time': np.round(header['EXPTIME'], 3),
            'ra': header['RA'],
            'dec': header['DEC'],
            'frame': header['FRAME'],
            'airmass': float(header['AIRMASS']),
            'detector_layout': detector_layout,
            'slit_length': slit_length,
            'slit_length_bins': np.ceil(slit_length/spatial_scale),
            'slit_width': slit_width,
            'slit_width_bins': np.ceil(slit_width/spectral_scale),
            'cross_disperser': _parse_cross_disperser(
                header['XDISPERS'].lower()),
            'cross_disperser_angle': np.round(header['XDANGL'], 5),
            'echelle_angle': np.round(header['ECHANGL'], 5),
            'spatial_binning': int(binning[0]),
            'spatial_bin_scale': spatial_scale,
            'spectral_binning': int(binning[1]),
            'spectral_bin_scale': spectral_scale,
            'pixel_size': pixel_size,
            'sky_position_angle': skypa,
        }


def _determine_gains(gain: str) -> list[float]:
    """
    Get list of gain values [e⁻/DN] for low or high gain.
    """
    if gain == 'low':
        gains = low_gain
    elif gain == 'high':
        gains = high_gain
    else:
        raise Exception('Gain must be supplied as a string. Choices are '
                        '"high" or "low". The HIRES default setting is '
                        '"low".')
    return gains


def _get_image_data(file_path: Path,
                    gain: str or None) -> u.Quantity:
    """
    Combine mosaic image data into a single array with proper inter-detector
    spacing, and multiply each detector image by its corresponding gain so they
    are comparable. Also block out the last 48 columns so they don't interfere
    with retrievals later on.
    """
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        binning = np.array(header['BINNING'].split(',')).astype(int)
        if len(hdul) > 1:  # post-2004 mosaic detectors
            # empirically-derived horizontal offset between mosaic detectors
            det_hoffsets = np.round(np.array([4, 7, 0])/binning[1]).astype(int)
            max_hoffset = np.max(det_hoffsets)
            gap01 = np.full(
                (int(detector_vertical_gaps[0]/binning[0]), 4096),
                fill_value=0)
            gap12 = np.full(
                (int(detector_vertical_gaps[1]/binning[0]), 4096),
                fill_value=0)
            images = []
            if gain is None:
                try:
                    gains = [header[f'CCDGN0{detector_number}']
                             for detector_number in range(1, 4)]
                except KeyError:
                    raise Exception(
                        "If using files with non-unique filenames, "
                        "e.g., hires00001.fits, you must specify the "
                        "detector gain as a string ('high' or 'low') "
                        "when instantiating the HIRESPipeline class.")
            else:
                gains = _determine_gains(gain)
            for detector_number in [1, 2, 3]:
                image_header = hdul[detector_number].header
                hoffset = det_hoffsets[detector_number-1]
                gain = gains[detector_number-1]
                detector_slice = _parse_mosaic_detector_slice(
                    image_header['DATASEC'])
                detector_image = np.flipud(
                    hdul[detector_number].data.T[detector_slice].astype(float))
                detector_image *= gain
                shape = detector_image.shape
                offset_image = np.full((shape[0], shape[1]+max_hoffset),
                                       fill_value=0)
                offset_image[:, hoffset:hoffset+shape[1]] = detector_image
                images.append(offset_image)
            data_image = np.concatenate(
                (images[0][:, max_hoffset:], gap01, images[1][:, max_hoffset:],
                 gap12, images[2][:, max_hoffset:]), axis=0)
            data_image[:, -bad_columns:] = 0
        else:  # pre-2004 single detector
            slice0 = header['PREPIX']
            slice1 = header['NAXIS1'] - header['POSTPIX']
            data_image = hdul[0].data[:, slice0:slice1]
    return data_image * u.electron


def _get_images_from_directory(directory: Path,
                               slit_length: float,
                               slit_width: float,
                               spatial_binning: float,
                               spectral_binning: float,
                               gain: str,
                               remove_cosmic_rays: bool,
                               log_path: Path) -> list[CCDData]:
    """
    Make a list of CCDData objects of combined mosaic data.
    """
    files = sorted(directory.glob('*.fits*'))
    n = len(files)
    images = []
    for i, file in enumerate(files):
        _log(log_path, f'         {i + 1}/{n}: {file.name}', new_line=False)
        header = _get_header(file, slit_length=slit_length,
                             slit_width=slit_width,
                             spatial_binning=spatial_binning,
                             spectral_binning=spectral_binning)
        data = CCDData(_get_image_data(file, gain=gain), header=header)
        if remove_cosmic_rays:
            data = ccdproc.cosmicray_lacosmic(data)
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


def _make_master_bias(file_directory: Path,
                      slit_length: float,
                      slit_width: float,
                      spatial_binning: float,
                      spectral_binning: float,
                      gain: str,
                      log_path: Path) -> CCDData:
    """
    Wrapper function to make a master bias detector image.
    """
    bias_images = _get_images_from_directory(Path(file_directory, 'bias'),
                                             slit_length=slit_length,
                                             slit_width=slit_width,
                                             spatial_binning=spatial_binning,
                                             spectral_binning=spectral_binning,
                                             gain=gain,
                                             remove_cosmic_rays=False,
                                             log_path=log_path)
    master_bias = _make_median_bias(bias_images)
    return master_bias


def _make_master_flux(file_directory: Path,
                      flux_type: str,
                      master_bias: CCDData,
                      slit_length: float,
                      slit_width: float,
                      spatial_binning: float,
                      spectral_binning: float,
                      gain: str,
                      log_path: Path) -> CCDData:
    """
    Wrapper function to make a master flat or arc detector image.
    """
    flux_images = _get_images_from_directory(Path(file_directory, flux_type),
                                             slit_length=slit_length,
                                             slit_width=slit_width,
                                             spatial_binning=spatial_binning,
                                             spectral_binning=spectral_binning,
                                             gain=gain,
                                             remove_cosmic_rays=False,
                                             log_path=log_path)
    flux_images = [ccdproc.subtract_bias(image, master=master_bias)
                   for image in flux_images]
    master_flux = _make_median_flux(flux_images)
    return master_flux


def _make_master_trace(file_directory: Path,
                       master_bias: CCDData,
                       slit_length: float,
                       slit_width: float,
                       spatial_binning: float,
                       spectral_binning: float,
                       gain: str,
                       log_path: Path) -> CCDData:
    """
    Wrapper function to load the first trace file and make a master order trace
    image.
    """
    trace_image = _get_images_from_directory(Path(file_directory, 'trace'),
                                             slit_length=slit_length,
                                             slit_width=slit_width,
                                             spatial_binning=spatial_binning,
                                             spectral_binning=spectral_binning,
                                             gain=gain,
                                             remove_cosmic_rays=False,
                                             log_path=log_path)[0]
    master_trace = ccdproc.subtract_bias(trace_image, master=master_bias)
    return master_trace


def _process_science_data(file_directory: Path,
                          sub_directory: str,
                          master_bias: CCDData,
                          master_flat: CCDData,
                          order_bounds: _OrderBounds,
                          wavelength_solution: _WavelengthSolution,
                          slit_length: float,
                          slit_width: float,
                          spatial_binning: float,
                          spectral_binning: float,
                          gain: str,
                          extinction_correct: bool,
                          log_path: Path) -> ([CCDData], [str]):
    """
    Wrapper function to apply all of the reduction steps to science data in a
    supplied directory.
    """
    _log(log_path, f'      Loading data and removing cosmic rays...')
    science_images = _get_images_from_directory(
        Path(file_directory, sub_directory),
        slit_length=slit_length,
        slit_width=slit_width,
        spatial_binning=spatial_binning,
        spectral_binning=spectral_binning,
        gain=gain,
        remove_cosmic_rays=True,
        log_path=log_path)
    count = len(science_images)
    reduced_science_images = []
    filenames = []
    _log(log_path, '      Reducing data...')
    for i, ccd_image in enumerate(science_images):
        filename = ccd_image.header['file_name']
        _log(log_path, f'         {i + 1}/{count}: {filename}', new_line=False)
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
            if extinction_correct:
                extinction_corrected_data = _extinction_correct(
                    rectified_data=flux_data,
                    wavelength_solution=wavelength_solution)
                data = extinction_corrected_data.data
                uncertainty = extinction_corrected_data.uncertainty.array
            else:
                data = flux_data.data
                uncertainty = flux_data.uncertainty.array
            ind = np.isnan(data)
            data[ind] = 0
            uncertainty[ind] = 0
            flux_data.data = data
            flux_data.uncertainty.array = uncertainty
            reduced_science_images.append(flux_data)
            filenames.append(filename)
    return reduced_science_images, filenames
