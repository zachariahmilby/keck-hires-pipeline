import warnings
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.nddata import CCDData
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from hirespipeline.files import make_directory
from hirespipeline.general import naif_codes, _log, _make_log
from hirespipeline.graphics import (rcparams, turn_off_axes, calculate_norm,
                                    bias_cmap, flux_cmap, nan_color)
from hirespipeline.image_processing import (_make_master_bias,
                                            _make_master_flux,
                                            _make_master_trace,
                                            _process_science_data)
from hirespipeline.order_tracing import _OrderTraces, _OrderBounds
from hirespipeline.saving import _save_as_fits
from hirespipeline.wavelength_solution import _WavelengthSolution


def stack_orders(rectified_data: np.ndarray,
                 dy=3) -> np.ndarray:
    """
    Stack individual order arrays into a single array.
    """
    n_orders, n_spa, n_spe = rectified_data.shape
    stacked_data = np.full(
        (int(n_orders * n_spa + (n_orders - 1) * dy), n_spe),
        fill_value=np.nan)
    for i in range(n_orders):
        s_ = np.s_[i*(n_spa+dy):i*(n_spa+dy)+n_spa]
        stacked_data[s_] = rectified_data[i]
    stacked_data[np.where(np.isnan(stacked_data))] = np.nan
    return stacked_data


# noinspection DuplicatedCode
def _calibration_qa_graphic(rectified_data: CCDData,
                            cmap: colors.Colormap,
                            savename: Path) -> None:
    """
    Generate a quality assurance graphic for calibration data like bias, arc
    or flats displaying data and uncertainty in physical units.
    """
    with plt.style.context(rcparams):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                                 constrained_layout=True, sharex='all',
                                 sharey='all', clear=True)
        [axis.set_facecolor(nan_color) for axis in axes.ravel()]
        [turn_off_axes(axis) for axis in axes.ravel()]
        data = stack_orders(rectified_data.data)
        img0 = axes[0].pcolormesh(
            data, cmap=cmap, norm=calculate_norm(data), rasterized=True)
        unc = stack_orders(rectified_data.uncertainty.array)
        img1 = axes[1].pcolormesh(
            unc, cmap=cmap, norm=calculate_norm(unc), rasterized=True)
        plt.colorbar(img0, ax=axes[0], label=f'{rectified_data.unit}')
        axes[0].set_title('Data')
        plt.colorbar(img1, ax=axes[1],
                     label=f'{rectified_data.uncertainty.unit}')
        axes[1].set_title('Uncertainty')
        make_directory(savename.parent)
        plt.savefig(savename)
        plt.close(fig)


# noinspection DuplicatedCode
def _science_qa_graphic(rectified_data: CCDData,
                        cmap: colors.Colormap,
                        savename: Path) -> None:
    """
    Generate a quality assurance graphic for science data displaying data and
    uncertainty in physical units.
    """
    with plt.style.context(rcparams):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                                 constrained_layout=True, sharex='all',
                                 sharey='all', clear=True)
        [axis.set_facecolor(nan_color) for axis in axes.ravel()]
        [turn_off_axes(axis) for axis in axes.ravel()]
        data = stack_orders(rectified_data.data)
        unc = stack_orders(rectified_data.uncertainty.array)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            snr = data / unc
        img0 = axes[0].pcolormesh(data, cmap=cmap,
                                  norm=calculate_norm(data, percentile=95),
                                  rasterized=True)
        plt.colorbar(img0, ax=axes[0], label=f'{rectified_data.unit}')
        img1 = axes[1].pcolormesh(
            unc, cmap=cmap,
            norm=calculate_norm(unc, percentile=95),
            rasterized=True)
        plt.colorbar(img1, ax=axes[1], label=f'{rectified_data.unit}')
        img2 = axes[2].pcolormesh(snr, cmap=cmap,
                                  norm=calculate_norm(snr, 95),
                                  rasterized=True)
        plt.colorbar(img2, ax=axes[2], label='Ratio')
        axes[0].set_title('Data')
        axes[1].set_title('Uncertainty')
        axes[2].set_title('Signal-to-Noise')
        make_directory(savename.parent)
        plt.savefig(savename)
        plt.close(fig)


class HIRESPipeline:
    """
    HIRES data reduction pipeline.
    """
    def __init__(self,
                 target: str or list[str],
                 file_directory: str or Path,
                 science_subdirectory: str or [str] = 'science',
                 slit_length: int | float = None,
                 slit_width: int | float = None,
                 spatial_binning: int = None,
                 spectral_binning: int = None,
                 gain: str = None):
        """
        Parameters
        ----------
        target: str or list(str)
            The targeted science body/bodies.
        file_directory : str or Path
            Parent directory with subdirectories containing bias, flat, arc and
            science frames.
        science_subdirectory : str or list(str)
            The name of the science file directory. It defaults to "science",
            but you may have multiple subsets for a given set of calibration
            files, so you can also specify individual locations for your
            science data.
        slit_length : float (optional)
            The length of the slit in arcseconds. Only necessary if using data
            with non-unique filenames, e.g., hires00001.fits.
        slit_width : float (optional)
            The width of the slit in arcseconds. Only necessary if using data
            with non-unique filenames, e.g., hires00001.fits.
        spatial_binning : int (optional)
            The number of spatial pixels per bin (probably 2 or 3). Only
            necessary if using data with non-unique filenames, e.g.,
            hires00001.fits.
        spectral_binning : int (optional)
            The number of spectra pixels per bin (almost certainly 1). Only
            necessary if using data with non-unique filenames, e.g.,
            hires00001.fits.
        gain : str (optional)
            The gain setting used, either 'low' (the HIRES default) or 'high'.
            Only necessary if using data with non-unique filenames, e.g.,
            hires00001.fits.
        """
        self._target = self._determine_input_type(target)
        self._file_directory = Path(file_directory)
        self._save_directory = make_directory(
            Path(Path(file_directory).parent, 'reduced'))
        self._science_subdirectory = self._determine_input_type(
            science_subdirectory)
        self._check_inputs()
        try:
            self._slit_length = float(slit_length)
            self._slit_width = float(slit_width)
            self._spatial_binning = float(spatial_binning)
            self._spectral_binning = float(spectral_binning)
            self._gain = gain
        except TypeError:
            self._slit_length = slit_length
            self._slit_width = slit_width
            self._spatial_binning = spatial_binning
            self._spectral_binning = spectral_binning
            self._gain = gain
        _make_log(self._save_directory)

    def log(self, string: str, new_line: bool = True) -> None:
        _log(self._save_directory, string, new_line=new_line)

    @staticmethod
    def _determine_input_type(science_subdirectory) -> list[str]:
        """
        Determines if you entered a single string for the subdirectory/target
        or a list of strings for multiple science subdirectories/targets.
        """
        if isinstance(science_subdirectory, list):
            return science_subdirectory
        elif isinstance(science_subdirectory, np.ndarray):
            return science_subdirectory.tolist()
        elif isinstance(science_subdirectory, str):
            return [science_subdirectory]
        else:
            raise Exception('Improper science subdirectory input type. Must be'
                            ' a string or a list of strings.')

    def _check_inputs(self) -> None:
        """
        Check if the number of subdirectories matches the number of targets.
        """
        if len(self._target) != len(self._science_subdirectory):
            raise Exception(f"Targets must match length of subdirectories. "
                            f"You've specified {self._target} for targets but "
                            f"{self._science_subdirectory} as subdirectories.")

    @staticmethod
    def _find_closest_target(targets: [str],
                             header: dict) -> np.ndarray:
        """
        Find Solar System target closest to telescope RA/Dec. Currently limited
        to just Jupiter and its Galilean satellites.
        """
        distances = []
        ref_coord = SkyCoord(ra=Angle(header['ra'], unit=u.hour),
                             dec=Angle(header['dec'], unit=u.degree))
        ext = ''
        if header['frame'] == 'apparent':
            ext = '_app'
        for target in targets:
            eph = Horizons(id=naif_codes[target], location='568',
                           epochs=Time(header['datetime']).jd).ephemerides()
            coord = SkyCoord(ra=Angle(eph[f'RA{ext}']),
                             dec=Angle(eph[f'DEC{ext}']))
            distances.append(ref_coord.separation(coord))
        closest = np.where(np.asarray(distances) == np.min(distances))[0][0]
        return np.asarray(targets)[closest]

    @staticmethod
    def _save_master_calibration_file(ccd_data: CCDData,
                                      order_numbers: np.ndarray,
                                      data_type: str,
                                      savepath: str or Path) -> None:
        """
        Save a master calibration FITS file.
        """
        _save_as_fits(data_header=ccd_data.header, data=ccd_data.data,
                      uncertainty=ccd_data.uncertainty.array,
                      unit=ccd_data.unit, data_type=data_type,
                      order_numbers=order_numbers, savepath=savepath)

    def _save_science_file(self,
                           reduced_data: CCDData,
                           target: str or [str],
                           wavelength_solution: _WavelengthSolution,
                           savepath: str or Path) -> None:
        """
        Save a science FITS file.
        """
        if isinstance(target, list):
            target = self._find_closest_target(target, reduced_data.header)
        _save_as_fits(data_header=reduced_data.header, data=reduced_data.data,
                      uncertainty=reduced_data.uncertainty.array,
                      unit=reduced_data.unit,
                      data_type='science', target=target,
                      order_numbers=wavelength_solution.order_numbers,
                      wavelength_centers=wavelength_solution.centers,
                      wavelength_edges=wavelength_solution.edges,
                      savepath=savepath)

    @staticmethod
    def _reduced_filename(original_filename: str) -> str:
        """
        Append '_reduced' to the end of a FITS filename.
        """
        return original_filename.replace('.fits', '_reduced.fits')

    # noinspection DuplicatedCode
    def run(self,
            save_graphics: bool = True,
            test_trace: bool = False,
            optimize_traces: bool = True,
            remove_airmass_extinction: bool = True):
        """
        Run the HIRES pipeline.

        Parameters
        ----------
        save_graphics : bool
            Whether or not to save a summary graphic along with the reduced
            data files. There are memory leaks in Matplotlib, and I've done by
            best, but if the pipeline ends up crashing it might be best to try
            to save files without summary graphics.

            Note: it will always save the trace and order edge graphics, since
            those are essential for quality assurance.
        test_trace: bool
            If you want to run the pipeline just far enough to test the order
            detection algorithm, set this to True. Mostly useful for my own
            debugging purposes.
        optimize_traces : bool
            Whether or not you want to optimize the traces. Setting to False
            improves speed and may prevent bad traces near detector edges.
            Default is True.
        remove_airmass_extinction: bool
            Whether or not you want to remove wavelength-dependent airmass
            extinction appropriate to the summit of Maunakea. Default is True.
            Uses the median curve in Figure 17 of Buton et al. (2003),
            doi:10.1051/0004-6361/201219834.
        """
        t0 = datetime.now(timezone.utc)
        print('')
        fmt = '%Y-%m-%d %H:%M:%S.%f'
        self.log(f"{datetime.now(timezone.utc).strftime(fmt)}")
        self.log(f'Running HIRES data reduction pipeline on directory '
                 f'"{str(self._file_directory)}"')

        # make master calibration detector images
        self.log('   Making master bias image...')
        master_bias = _make_master_bias(
            file_directory=self._file_directory,
            slit_length=self._slit_length,
            slit_width=self._slit_width,
            spatial_binning=self._spatial_binning,
            spectral_binning=self._spectral_binning,
            gain=self._gain,
            log_path=self._save_directory)

        self.log('   Making master flat image...')
        master_flat = _make_master_flux(
            file_directory=self._file_directory,
            flux_type='flat',
            master_bias=master_bias,
            slit_length=self._slit_length,
            slit_width=self._slit_width,
            spatial_binning=self._spatial_binning,
            spectral_binning=self._spectral_binning,
            gain=self._gain,
            log_path=self._save_directory)

        self.log('   Making master arc image...')
        master_arc = _make_master_flux(file_directory=self._file_directory,
                                       flux_type='arc',
                                       master_bias=master_bias,
                                       slit_length=self._slit_length,
                                       slit_width=self._slit_width,
                                       spatial_binning=self._spatial_binning,
                                       spectral_binning=self._spectral_binning,
                                       gain=self._gain,
                                       log_path=self._save_directory)

        self.log('   Making master trace image...')
        master_trace = _make_master_trace(
            file_directory=self._file_directory,
            master_bias=master_bias,
            slit_length=self._slit_length,
            slit_width=self._slit_width,
            spatial_binning=self._spatial_binning,
            spectral_binning=self._spectral_binning,
            gain=self._gain,
            log_path=self._save_directory)
        self.log('   Tracing echelle orders...')
        order_traces = _OrderTraces(master_trace=master_trace,
                                    log_path=self._save_directory,
                                    optimize=optimize_traces)
        order_traces.quality_assurance(Path(self._save_directory))

        if not test_trace:
            self.log('   Finding order edges...')
            order_bounds = _OrderBounds(order_traces=order_traces,
                                        master_flat=master_flat,
                                        log_path=self._save_directory)
            order_bounds.quality_assurance(Path(self._save_directory))

            self.log('   Calculating wavelength solution...')
            wavelength_solution = _WavelengthSolution(
                master_arc=master_arc,
                master_flat=master_flat,
                order_bounds=order_bounds,
                log_path=self._save_directory)
            wavelength_solution.quality_assurance(Path(self._save_directory))

            self.log('   Rectifying master bias...')
            rectified_master_bias = order_bounds.rectify(master_bias)
            self._save_master_calibration_file(
                ccd_data=rectified_master_bias,
                order_numbers=wavelength_solution.order_numbers,
                data_type='master bias',
                savepath=Path(self._save_directory, 'master_bias.fits.gz'))
            if save_graphics:
                _calibration_qa_graphic(
                    rectified_data=rectified_master_bias,
                    cmap=bias_cmap(),
                    savename=Path(self._save_directory, 'quality_assurance',
                                  'master_bias.jpg'))

            self.log('   Rectifying master flat...')
            rectified_master_flat = order_bounds.rectify(master_flat)
            self._save_master_calibration_file(
                ccd_data=rectified_master_flat,
                order_numbers=wavelength_solution.order_numbers,
                data_type='master flat',
                savepath=Path(self._save_directory, 'master_flat.fits.gz'))
            if save_graphics:
                _calibration_qa_graphic(
                    rectified_data=rectified_master_flat,
                    cmap=flux_cmap(),
                    savename=Path(self._save_directory, 'quality_assurance',
                                  'master_flat.jpg'))

            self.log('   Rectifying master arc...')
            rectified_master_arc = order_bounds.rectify(master_arc)
            self._save_master_calibration_file(
                ccd_data=rectified_master_arc,
                order_numbers=wavelength_solution.order_numbers,
                data_type='master arc',
                savepath=Path(self._save_directory, 'master_arc.fits.gz'))
            if save_graphics:
                _calibration_qa_graphic(
                    rectified_data=rectified_master_arc,
                    cmap=flux_cmap(),
                    savename=Path(self._save_directory, 'quality_assurance',
                                  'master_arc.jpg'))

            for target, sub_directory in zip(self._target,
                                             self._science_subdirectory):
                self.log(f'   Processing data in "{sub_directory}" '
                         f'directory...')

                science_images, filenames = _process_science_data(
                    file_directory=self._file_directory,
                    sub_directory=sub_directory,
                    master_bias=rectified_master_bias,
                    master_flat=rectified_master_flat,
                    order_bounds=order_bounds,
                    wavelength_solution=wavelength_solution,
                    slit_length=self._slit_length,
                    slit_width=self._slit_width,
                    spatial_binning=self._spatial_binning,
                    spectral_binning=self._spectral_binning,
                    gain=self._gain,
                    extinction_correct=remove_airmass_extinction,
                    log_path=self._save_directory)
                self.log(f'      Saving data...')
                count = len(science_images)
                for i, (image, filename) in enumerate(
                        zip(science_images, filenames)):
                    if filename.split('.')[-1] == 'gz':
                        ext = '.fits.gz'
                    else:
                        ext = '.fits'
                    self.log(f'         {i + 1}/{count}: {filename}',
                             new_line=False)
                    self._save_science_file(
                        reduced_data=image,
                        target=target,
                        wavelength_solution=wavelength_solution,
                        savepath=Path(self._save_directory, sub_directory,
                                      filename.replace(ext, f'_reduced{ext}')))
                    if save_graphics:
                        _science_qa_graphic(
                            rectified_data=image,
                            cmap=flux_cmap(),
                            savename=Path(self._save_directory, sub_directory,
                                          filename.replace(ext, '_reduced.jpg')
                                          ))

        elapsed_time = datetime.now(timezone.utc) - t0
        self.log(f'Reduction complete, time elapsed: {elapsed_time}.')
