import warnings
from datetime import datetime, timezone
from pathlib import Path

import ccdproc
import numpy as np
from astropy.nddata import CCDData

from hirespipeline.files import _make_directory
from hirespipeline.general import readnoise
from hirespipeline.image_processing import _combine_mosaic_image, \
    _get_header, _make_median_image
from hirespipeline.order_tracing import _OrderTraces, _OrderBounds
from hirespipeline.saving import _save_as_fits
from hirespipeline.wavelength_solution import _WavelengthSolution


def _get_images_from_directory(
        directory: Path, remove_cosmic_rays: bool = False) -> list[CCDData]:
    """
    Make a list of CCDData objects of combined mosaic data.
    """
    files = sorted(directory.glob('*.fits*'))
    images = []
    for file in files:
        header = _get_header(file)
        data = CCDData(_combine_mosaic_image(file), header=header)
        if remove_cosmic_rays:
            data = ccdproc.cosmicray_lacosmic(data)
        data_with_uncertainty = ccdproc.create_deviation(
            data, readnoise=readnoise, disregard_nan=True)
        images.append(data_with_uncertainty)
    return images


class HIRESPipeline:

    def __init__(self, target: str or list[str], file_directory: str or Path,
                 science_subdirectory: str or [str] = 'science'):
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
        """
        self._target = self._determine_input_type(target)
        self._file_directory = Path(file_directory)
        self._save_directory = _make_directory(
            Path(Path(file_directory).parent, 'reduced'))
        self._science_subdirectory = self._determine_input_type(
            science_subdirectory)

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
            ccd_data: CCDData, order_bounds: _OrderBounds, data_type: str,
            order_numbers: np.ndarray, savepath: str or Path):
        rectified_data = order_bounds.rectify_data(ccd_data.data)
        rectified_uncertainty = order_bounds.rectify_data(
            ccd_data.uncertainty.array)
        _save_as_fits(data_header=ccd_data.header, data=rectified_data,
                      uncertainty=rectified_uncertainty, unit=ccd_data.unit,
                      data_type=data_type, order_numbers=order_numbers,
                      savepath=savepath)

    @staticmethod
    def _save_science_file(
            reduced_data: CCDData, raw_data: CCDData,
            order_bounds: _OrderBounds, target: str,
            wavelength_solution: _WavelengthSolution, savepath: str or Path):
        rectified_data = order_bounds.rectify_data(reduced_data.data)
        rectified_uncertainty = order_bounds.rectify_data(
            reduced_data.uncertainty.array)
        rectified_raw_data = order_bounds.rectify_data(raw_data.data)
        rectified_raw_uncertainty = order_bounds.rectify_data(
            raw_data.uncertainty.array)
        _save_as_fits(data_header=reduced_data.header, data=rectified_data,
                      uncertainty=rectified_uncertainty,
                      raw_data=rectified_raw_data,
                      raw_uncertainty=rectified_raw_uncertainty,
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

        print('   Making master bias...')
        bias_images = _get_images_from_directory(
            Path(self._file_directory, 'bias'))
        master_bias = _make_median_image(bias_images)

        print('   Making master flat...')
        flat_images = _get_images_from_directory(
            Path(self._file_directory, 'flat'))
        master_flat = _make_median_image(flat_images)
        master_flat = ccdproc.subtract_bias(master_flat, master=master_bias)

        print('   Making master arc...')
        arc_images = _get_images_from_directory(
            Path(self._file_directory, 'arc'))
        master_arc = _make_median_image(arc_images)
        master_arc = ccdproc.subtract_bias(master_arc, master=master_bias)

        print('   Tracing echelle orders...')
        master_trace = CCDData(
            _combine_mosaic_image(
                sorted(Path(self._file_directory, 'trace').glob('*fits*'))[0]))
        master_trace = ccdproc.subtract_bias(master_trace, master=master_bias)
        order_traces = _OrderTraces(master_trace=master_trace)
        order_bounds = _OrderBounds(order_traces=order_traces,
                                    master_flat=master_flat)

        print('   Calculating wavelength solution...')
        wavelength_solution = _WavelengthSolution(
            master_arc=master_arc, master_flat=master_flat,
            order_bounds=order_bounds)

        print('   Saving calibration data...')
        self._save_master_calibration_file(
            ccd_data=master_bias, data_type='master bias',
            order_bounds=order_bounds,
            order_numbers=wavelength_solution.order_numbers,
            savepath=Path(self._save_directory, 'master_bias.fits.gz'))
        self._save_master_calibration_file(
            ccd_data=master_flat, data_type='master flat',
            order_bounds=order_bounds,
            order_numbers=wavelength_solution.order_numbers,
            savepath=Path(self._save_directory, 'master_flat.fits.gz'))
        self._save_master_calibration_file(
            ccd_data=master_arc, data_type='master arc',
            order_bounds=order_bounds,
            order_numbers=wavelength_solution.order_numbers,
            savepath=Path(self._save_directory, 'master_arc.fits.gz'))

        print('   Saving quality assurance graphics...')
        qa_path = _make_directory(
            Path(self._save_directory, 'quality_assurance'))
        order_traces.quality_assurance(qa_path)
        order_bounds.quality_assurance(qa_path)
        wavelength_solution.quality_assurance(qa_path)

        for target, directory in zip(self._target, self._science_subdirectory):
            print(f'   Reducing data in directory "{directory}"...')
            science_images = _get_images_from_directory(
                Path(self._file_directory, directory), remove_cosmic_rays=True)
            savepath = _make_directory(Path(self._save_directory, directory))
            count = len(science_images)
            for i, ccd_image in enumerate(science_images):
                filename = ccd_image.header['file_name']
                print(f"      {i + 1}/{count}: {filename}")
                bias_subtracted_data = ccdproc.subtract_bias(
                    ccd_image, master=master_bias)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    flat_corrected_data = ccdproc.flat_correct(
                        bias_subtracted_data, flat=master_flat,
                        norm_value=np.nanmean(master_flat.data))
                reduced_filename = self._reduced_filename(filename)
                self._save_science_file(
                    reduced_data=flat_corrected_data, raw_data=ccd_image,
                    order_bounds=order_bounds,
                    target=target, wavelength_solution=wavelength_solution,
                    savepath=Path(savepath, reduced_filename))

        print(f'Processing complete, time elapsed '
              f'{datetime.now(timezone.utc) - t0}.')
