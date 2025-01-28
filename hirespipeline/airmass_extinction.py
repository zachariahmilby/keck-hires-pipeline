from pathlib import Path

import numpy as np
from astropy.nddata import CCDData, StdDevUncertainty

from hirespipeline.general import package_directory
from hirespipeline.wavelength_solution import _WavelengthSolution


def _load_extinction_data() -> dict:
    """
    Load Buton et al. (2003) Maunakea extinction data.
    """
    ext_path = Path(package_directory, 'anc',
                    'mauna_kea_airmass_extinction.dat')
    wavelength, factor = np.genfromtxt(ext_path, unpack=True,
                                       skip_header=True, delimiter=' ')
    return {'wavelength': wavelength,
            'factor': factor}


def _extinction_correction(airmass: float,
                           wavelengths: np.ndarray,
                           rectified_data: np.ndarray) -> np.ndarray:
    """
    Apply airmass-extinction correction.
    """
    extinction_data = _load_extinction_data()
    extinction_corrected_data = np.zeros_like(rectified_data)
    for (order, data) in enumerate(rectified_data):
        data = np.asarray(data)
        interp_extinction = np.interp(
            wavelengths[order], extinction_data['wavelength'],
            extinction_data['factor'])
        factor = airmass / 100 ** (1 / 5)
        extinction = np.tile(10 ** (interp_extinction * factor),
                             (data.shape[0], 1))
        extinction_corrected_data[order] = data * extinction
    return extinction_corrected_data


def _extinction_correct(rectified_data: CCDData,
                        wavelength_solution: _WavelengthSolution) -> CCDData:
    """
    Wrapper function to apply extinction correction to a CCDData object.
    """
    airmass = rectified_data.header['airmass']
    data = rectified_data.data
    unc = rectified_data.uncertainty.array
    wavelengths = wavelength_solution.centers
    extinction_corrected_data = _extinction_correction(
        airmass=airmass, wavelengths=wavelengths,
        rectified_data=data)
    extinction_corrected_unc = _extinction_correction(
        airmass=airmass, wavelengths=wavelengths, rectified_data=unc)
    return CCDData(data=extinction_corrected_data,
                   uncertainty=StdDevUncertainty(extinction_corrected_unc),
                   unit=rectified_data.unit,
                   header=rectified_data.header.copy())
