from pathlib import Path
from datetime import datetime, timezone

from hirespipeline.files import _Files, _make_directory
from hirespipeline.reduction import _MasterBias, _MasterFlat, _MasterArc, \
    _MasterTrace, _OrderBounds, _WavelengthSolution, _ScienceData


class HIRESPipeline:

    def __init__(self, target: str, file_directory: str,
                 science_file_directory: str = 'science'):
        """
        Use this class to run the pipeline.

        Parameters
        ----------
        file_directory : str
            Parent directory with subdirectories containing bias, flat, arc and
            science frames.
        science_file_directory : str
            The name of the science file directory. It defaults to "science",
            but you may have multiple subsets for a given set of calibration
            files, so you can also specify individual locations for your
            science data.
        """
        self._target = target
        self._file_directory = Path(file_directory)
        self._save_directory = _make_directory(
            Path(Path(file_directory).parent, 'reduced'))
        self._science_file_directory = science_file_directory

    def run(self):
        t0 = datetime.now(timezone.utc)
        print('Running HIRES data reduction pipeline.')
        # ccd = _retrieve_ccd_data_from_fits(file_path=science_file,
        #                                    remove_cosmic_rays=True)

        # make master calibration images
        print('   Making master bias...', end='\r')
        master_bias = _MasterBias(self._file_directory)

        print('   Making master flat...', end='\r')
        master_flat = _MasterFlat(self._file_directory, master_bias)

        print('   Making master arc...', end='\r')
        master_arc = _MasterArc(self._file_directory, master_bias, master_flat)

        print('   Tracing echelle orders...', end='\r')
        master_trace = _MasterTrace(self._file_directory, master_bias,
                                    master_flat)

        # find order bounds
        print('   Finding echelle order bounds...', end='\r')
        order_bounds = _OrderBounds(master_trace, master_flat)

        # calcualte wavelength solution
        print('   Calculating wavelength solution...', end='\r')
        wavelength_solution = _WavelengthSolution(master_arc, master_flat,
                                                  order_bounds)

        # save quality assurance products
        qa_path = _make_directory(
            Path(self._save_directory, 'quality_assurance'))
        master_trace.quality_assurance(qa_path)
        order_bounds.quality_assurance(qa_path)
        wavelength_solution.quality_assurance(qa_path)

        # reduce science data
        science_files = _Files(Path(self._file_directory,
                                    self._science_file_directory))
        n = len(science_files.paths)
        for i, file in enumerate(science_files.paths):
            print(f'      Reducing science image {i+1}/{n}: {file.name}...',
                  end='\r')
            science_data = _ScienceData(
                self._target, file, self._save_directory,
                self._science_file_directory, master_bias, master_flat,
                master_arc, order_bounds, wavelength_solution)
            science_data.reduce_science_data()

        print(f'Processing complete, time elapsed '
              f'{datetime.now(timezone.utc) - t0}.')
