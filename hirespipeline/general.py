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
