from pathlib import Path

import astropy.units as u

package_directory = Path(__file__).resolve().parent

airmass = u.def_unit('airmass', represents=u.dimensionless_unscaled)

shift_params = dict(order=1, prefilter=False)

# HIRES properties
readnoise = 3 * u.electron
darkcurrent = 2 * u.electron / u.pixel / u.hour
low_gain = [1.95, 2.09, 2.09]
high_gain = [0.78, 0.84, 0.89]
detector_vertical_gaps = [36, 48]  # empirically-derived
detectors_horizontal_offset = [0, 2.5, -1.5]
bad_columns = 50  # number of last columns to eliminate

naif_codes = {'Jupiter': '599', 'Io': '501', 'Europa': '502',
              'Ganymede': '503', 'Callisto': '504', 'Maunakea': '568'}


def _make_log(path: Path):
    with open(Path(path, 'log.txt'), 'w') as _:
        pass


def _write_log(path: Path, string: str):
    with open(Path(path, 'log.txt'), 'a') as file:
        file.write(string + '\n')


def _log(path, string, silent: bool = False):
    _write_log(path, string)
    if not silent:
        print(string)


def air_to_vac(wavelength: u.Quantity) -> u.Quantity:
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006. Taken from `specutils`.
    """
    wlum = wavelength.to(u.um).value
    return (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wavelength


def vac_to_air(wavelength: u.Quantity) -> u.Quantity:
    """
    Griesen 2006 reports that the error in naively inverting Eqn 65 is less
    than 10^-9 and therefore acceptable.  This is therefore eqn 67. Taken from
    `specutils'.
    """
    wlum = wavelength.to(u.um).value
    nl = (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4))
    return wavelength/nl
