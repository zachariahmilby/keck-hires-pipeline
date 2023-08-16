from pathlib import Path

import astropy.units as u

package_directory = Path(__file__).resolve().parent

airmass = u.def_unit('airmass', represents=u.dimensionless_unscaled)

# HIRES properties
readnoise = 3 * u.electron
darkcurrent = 2 * u.electron / u.pixel / u.hour
low_gain = [1.95, 2.09, 2.09]
high_gain = [0.78, 0.84, 0.89]
detector_spacing_pixels = [0, 41, 53]  # empirically-derived
