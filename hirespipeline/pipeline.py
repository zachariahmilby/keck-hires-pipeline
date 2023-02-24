import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import CCDData

from hirespipeline.files import make_directory
from hirespipeline.graphics import rcparams, turn_off_axes, calculate_norm, \
    bias_cmap, flux_cmap
from hirespipeline.image_processing import _make_master_bias, \
    _make_master_flux, _make_master_trace, _process_science_data
from hirespipeline.order_tracing import _OrderTraces, _OrderBounds
from hirespipeline.saving import _save_as_fits
from hirespipeline.wavelength_solution import _WavelengthSolution


def stack_orders(rectified_data: np.ndarray, dy=3):
    n_orders, n_spa, n_spe = rectified_data.shape
    stacked_data = np.full(
        (int(n_orders * n_spa + (n_orders - 1) * dy), n_spe),
        fill_value=np.nan)
    for i in range(n_orders):
        stacked_data[i*(n_spa + dy):i*(n_spa + dy)+n_spa] = rectified_data[i]
    return stacked_data


def _calibration_qa_graphic(rectified_data: CCDData,
                            cmap: colors.Colormap, savename: Path):
    with plt.style.context(rcparams):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                                 constrained_layout=True, sharex='all',
                                 sharey='all')
        [turn_off_axes(axis) for axis in axes.ravel()]
        img0 = axes[0].pcolormesh(stack_orders(rectified_data.data),
                                  cmap=cmap,
                                  norm=calculate_norm(rectified_data.data),
                                  rasterized=True)
        img1 = axes[1].pcolormesh(
            stack_orders(rectified_data.uncertainty.array), cmap=cmap,
            norm=calculate_norm(rectified_data.uncertainty.array, ),
            rasterized=True)
        plt.colorbar(img0, ax=axes[0], label=f'{rectified_data.unit}')
        axes[0].set_title('Data')
        plt.colorbar(img1, ax=axes[1],
                     label=f'{rectified_data.uncertainty.unit}')
        axes[1].set_title('Uncertainty')
        make_directory(savename.parent)
        plt.savefig(savename)
        plt.close(fig)


def _science_qa_graphic(rectified_data: CCDData,
                        cmap: colors.Colormap, savename: Path):
    with plt.style.context(rcparams):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                                 constrained_layout=True, sharex='all',
                                 sharey='all')
        [turn_off_axes(axis) for axis in axes.ravel()]
        data = stack_orders(rectified_data.data)
        unc = stack_orders(rectified_data.uncertainty.array)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            snr = data/unc
        img0 = axes[0].pcolormesh(data, cmap=cmap,
                                  norm=calculate_norm(rectified_data.data,
                                                      percentile=95),
                                  rasterized=True)
        plt.colorbar(img0, ax=axes[0], label=f'{rectified_data.unit}')
        img1 = axes[1].pcolormesh(
            unc, cmap=cmap,
            norm=calculate_norm(rectified_data.uncertainty.array,
                                percentile=95),
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

    def __init__(self, target: str or list[str], file_directory: str or Path,
                 science_subdirectory: str or [str] = 'science',
                 slit_length: int | float = None,
                 slit_width: int | float = None, spatial_binning: int = None,
                 spectral_binning: int = None, gain: str = None):
        """
        Use this class to run the pipeline.

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

    @staticmethod
    def _determine_input_type(science_subdirectory):
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

    @staticmethod
    def _save_master_calibration_file(
            ccd_data: CCDData, order_numbers: np.ndarray, data_type: str,
            savepath: str or Path):
        _save_as_fits(data_header=ccd_data.header, data=ccd_data.data,
                      uncertainty=ccd_data.uncertainty.array,
                      unit=ccd_data.unit, data_type=data_type,
                      order_numbers=order_numbers, savepath=savepath)

    @staticmethod
    def _save_science_file(
            reduced_data: CCDData, target: str,
            wavelength_solution: _WavelengthSolution, savepath: str or Path):
        _save_as_fits(data_header=reduced_data.header, data=reduced_data.data,
                      uncertainty=reduced_data.uncertainty.array,
                      unit=reduced_data.unit,
                      data_type='science', target=target,
                      order_numbers=wavelength_solution.order_numbers,
                      wavelength_centers=wavelength_solution.centers,
                      wavelength_edges=wavelength_solution.edges,
                      savepath=savepath)

    @staticmethod
    def _reduced_filename(original_filename: str):
        return original_filename.replace('.fits', '_reduced.fits')

    def run(self):
        t0 = datetime.now(timezone.utc)
        print(f'Running HIRES data reduction pipeline on '
              f'{str(self._file_directory)}')

        # make master calibration detector images
        print('   Making master bias image...')
        master_bias = _make_master_bias(
            file_directory=self._file_directory,
            slit_length=self._slit_length,
            slit_width=self._slit_width,
            spatial_binning=self._spatial_binning,
            spectral_binning=self._spectral_binning,
            gain=self._gain)

        print('   Making master flat image...')
        master_flat = _make_master_flux(
            file_directory=self._file_directory,
            flux_type='flat',
            master_bias=master_bias,
            slit_length=self._slit_length,
            slit_width=self._slit_width,
            spatial_binning=self._spatial_binning,
            spectral_binning=self._spectral_binning,
            gain=self._gain)

        print('   Making master arc image...')
        master_arc = _make_master_flux(file_directory=self._file_directory,
                                       flux_type='arc',
                                       master_bias=master_bias,
                                       slit_length=self._slit_length,
                                       slit_width=self._slit_width,
                                       spatial_binning=self._spatial_binning,
                                       spectral_binning=self._spectral_binning,
                                       gain=self._gain)

        print('   Tracing echelle orders...')
        master_trace = _make_master_trace(
            file_directory=self._file_directory, master_bias=master_bias,
            slit_length=self._slit_length,
            slit_width=self._slit_width,
            spatial_binning=self._spatial_binning,
            spectral_binning=self._spectral_binning,
            gain=self._gain)
        order_traces = _OrderTraces(master_trace=master_trace)
        order_traces.quality_assurance(Path(self._save_directory))
        order_bounds = _OrderBounds(order_traces=order_traces,
                                    master_flat=master_flat)
        order_bounds.quality_assurance(Path(self._save_directory))

        print('   Calculating wavelength solution...')
        wavelength_solution = _WavelengthSolution(master_arc=master_arc,
                                                  master_flat=master_flat,
                                                  order_bounds=order_bounds)
        wavelength_solution.quality_assurance(Path(self._save_directory))

        print('   Rectifying master bias...')
        rectified_master_bias = order_bounds.rectify(master_bias)
        self._save_master_calibration_file(
            ccd_data=rectified_master_bias,
            order_numbers=wavelength_solution.order_numbers,
            data_type='master bias',
            savepath=Path(self._save_directory, 'master_bias.fits.gz'))
        _calibration_qa_graphic(
            rectified_data=rectified_master_bias, cmap=bias_cmap(),
            savename=Path(self._save_directory, 'quality_assurance',
                          'master_bias.jpg'))

        print('   Rectifying master flat...')
        rectified_master_flat = order_bounds.rectify(master_flat)
        self._save_master_calibration_file(
            ccd_data=rectified_master_flat,
            order_numbers=wavelength_solution.order_numbers,
            data_type='master flat',
            savepath=Path(self._save_directory, 'master_flat.fits.gz'))
        _calibration_qa_graphic(
            rectified_data=rectified_master_flat, cmap=flux_cmap(),
            savename=Path(self._save_directory, 'quality_assurance',
                          'master_flat.jpg'))

        print('   Rectifying master arc...')
        rectified_master_arc = order_bounds.rectify(master_arc)
        self._save_master_calibration_file(
            ccd_data=rectified_master_arc,
            order_numbers=wavelength_solution.order_numbers,
            data_type='master arc',
            savepath=Path(self._save_directory, 'master_arc.fits.gz'))
        _calibration_qa_graphic(
            rectified_data=rectified_master_arc, cmap=flux_cmap(),
            savename=Path(self._save_directory, 'quality_assurance',
                          'master_arc.jpg'))

        for target, sub_directory in zip(self._target,
                                         self._science_subdirectory):
            print(f'   Reducing data in "{sub_directory}" directory...')
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
                gain=self._gain)
            for image, filename in zip(science_images, filenames):
                if filename.split('.')[-1] == 'gz':
                    ext = '.fits.gz'
                else:
                    ext = '.fits'
                self._save_science_file(
                    reduced_data=image, target=target,
                    wavelength_solution=wavelength_solution,
                    savepath=Path(self._save_directory, sub_directory,
                                  filename.replace(ext, f'_reduced{ext}')))
                _science_qa_graphic(
                    rectified_data=image, cmap=flux_cmap(),
                    savename=Path(self._save_directory, sub_directory,
                                  filename.replace(ext, '_reduced.jpg')))

        print(f'Processing complete, time elapsed '
              f'{datetime.now(timezone.utc) - t0}.')
