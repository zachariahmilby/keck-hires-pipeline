import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from hirespipeline.files import make_directory


def fix_nonprimary_header_2d(header: fits.Header, unit: str):
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
    header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
    header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
    header.append(('BUNIT', f'{unit}', 'data physical units'))


def fix_nonprimary_header_1d(header: fits.Header, unit: str):
    header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bin centers')
    header['NAXIS2'] = (header['NAXIS2'], 'number of echelle orders')
    header.append(('BUNIT', f'{unit}', 'data physical units'))


# noinspection PyTypeChecker, DuplicatedCode
def _save_as_fits(data_header: dict, data: np.ndarray, uncertainty: np.ndarray,
                  unit: str or None, data_type: str, savepath: Path,
                  target: str = None,
                  wavelength_centers: np.ndarray = None,
                  wavelength_edges: np.ndarray = None,
                  order_numbers: np.ndarray = None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=fits.verify.VerifyWarning)

        # data (primary extension)
        primary_hdu = fits.PrimaryHDU(data)
        header = primary_hdu.header
        header['NAXIS1'] = (header['NAXIS1'], 'number of spectral bins')
        header['NAXIS2'] = (header['NAXIS2'], 'number of spatial bins')
        header['NAXIS3'] = (header['NAXIS3'], 'number of echelle orders')
        header.append(('TELESCOP', 'Keck I', 'name of telescope'))
        header.append(('INSTRUME', 'HIRES', 'name of instrument'))
        header.append(('IMAGETYP', f'{data_type}', 'data type'))
        header.append(('BUNIT', f'{unit}', 'data physical units'))
        try:
            header.append(('OBJECT', f'{target}', 'name of target body'))
        except KeyError:
            pass
        try:
            header.append(('DATE-OBS', f"{data_header['datetime']}",
                           'UTC datetime at start of observation'))
        except KeyError:
            pass
        try:
            header.append(('EXPTIME', data_header['exposure_time'],
                           'exposure time [seconds]'))
        except KeyError:
            pass
        try:
            header.append(('AIRMASS', data_header['airmass'],
                           'airmass'))
        except KeyError:
            pass
        header.append(('OBSERVER', data_header['observers'],
                       'names of observers'))
        header.append(('SLITLEN', data_header['slit_length'],
                       'slit length [arcsec]'))
        header.append(('SLITLENB', data_header['slit_length_bins'],
                       'slit length [bins]'))
        header.append(('SLITWID', data_header['slit_width'],
                       'slit width [arcsec]'))
        header.append(('SLITWIDB', data_header['slit_width_bins'],
                       'slit width [bins]'))
        header.append(('XDNAME', data_header['cross_disperser'],
                       'name of cross diserpser'))
        header.append(('XDANG', data_header['cross_disperser_angle'],
                       'cross disperser angle [deg]'))
        header.append(('ECHANG', data_header['echelle_angle'],
                       'echelle angle [deg]'))
        header.append(('SPABIN', data_header['spatial_binning'],
                       'spatial binning [pix/bin]'))
        header.append(('SPEBIN', data_header['spectral_binning'],
                       'spectral binning [pix/bin]'))
        header.append(('SPASCALE', data_header['spatial_bin_scale'],
                       'spatial bin scale [arcsec/bin]'))
        header.append(('SPESCALE', data_header['spectral_bin_scale'],
                       'spectral bin scale [arcsec/bin]'))
        header.append(('PIXWIDTH', data_header['pixel_size'],
                       'pixel width [micron]'))
        header.append(('SKYPA', data_header['sky_position_angle'],
                       'slit rotation angle [deg]'))

        # data uncertainty
        primary_unc_hdu = fits.ImageHDU(uncertainty, name='PRIMARY_UNC')
        fix_nonprimary_header_2d(primary_unc_hdu.header, unit)

        # echelle orders
        if order_numbers is not None:
            echelle_orders_hdu = fits.ImageHDU(
                order_numbers, name='ECHELLE_ORDERS')
        else:
            echelle_orders_hdu = None

        # wavelength solution bin centers
        if wavelength_centers is not None:
            wavelength_centers_hdu = fits.ImageHDU(
                wavelength_centers, name='BIN_CENTER_WAVELENGTHS')
            fix_nonprimary_header_1d(wavelength_centers_hdu.header, 'nm')
        else:
            wavelength_centers_hdu = None

        # wavelength solution bin edges
        if wavelength_edges is not None:
            wavelength_edges_hdu = fits.ImageHDU(
                wavelength_edges, name='BIN_EDGE_WAVELENGTHS')
            fix_nonprimary_header_1d(wavelength_edges_hdu.header, 'nm')
        else:
            wavelength_edges_hdu = None

        hdus = [primary_hdu, primary_unc_hdu, echelle_orders_hdu,
                wavelength_centers_hdu, wavelength_edges_hdu]
        usable_hdus = [i for i in hdus if i is not None]
        hdul = fits.HDUList(usable_hdus)

        make_directory(savepath.parent)
        hdul.writeto(savepath, overwrite=True)
        hdul.close()
