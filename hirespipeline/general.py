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


def _make_log(path: Path) -> None:
    """
    Create a blank log file.
    """
    with open(Path(path, 'log.txt'), 'w') as _:
        pass


def _write_log(path: Path,
               string: str) -> None:
    """
    Write a string to a log file.
    """
    with open(Path(path, 'log.txt'), 'a') as file:
        file.write(string + '\n')


def _log(path, string,
         silent: bool = False,
         new_line: bool = True) -> None:
    """
    Wrapper function to log a string and optionally print it in the terminal.
    """
    _write_log(path, string)
    if not silent:
        print("\33[2K\r", end="")
        if not new_line:
            print(string, end='\r')
        else:
            print(string)


def air_to_vac(wavelength: u.Quantity) -> u.Quantity:
    """
    Implements the air-to-vacuum wavelength conversion described in
    equation (65) of Griesen et al. (2006), doi:10.1051/0004-6361:20053818.
    Copied from the implementation in `specutils`.
    """
    wlum = wavelength.to(u.um).value
    n = (1 + 1e-6 * (287.6155 + 1.62887/wlum**2 + 0.01360/wlum**4))
    return n * wavelength


def vac_to_air(wavelength: u.Quantity) -> u.Quantity:
    """
    Implements the vacuum-to-air wavelength conversion described in
    equation (67) of Griesen et al. (2006), doi:10.1051/0004-6361:20053818.
    Copied from the implementation in `specutils`.
    """
    wlum = wavelength.to(u.um).value
    n = (1 + 1e-6 * (287.6155 + 1.62887/wlum**2 + 0.01360/wlum**4))
    return wavelength / n
