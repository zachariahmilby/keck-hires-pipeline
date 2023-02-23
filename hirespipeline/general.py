from pathlib import Path

import astropy.units as u

package_directory = Path(__file__).resolve().parent

airmass = u.def_unit('airmass', represents=u.dimensionless_unscaled)

# HIRES properties
readnoise = 3 * u.electron
darkcurrent = 2 * u.electron / u.pixel / u.hour
