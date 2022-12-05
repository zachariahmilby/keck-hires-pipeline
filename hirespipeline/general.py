from pathlib import Path
import astropy.units as u

package_directory = Path(__file__).resolve().parent
airmass = u.def_unit('airmass', represents=u.dimensionless_unscaled)
