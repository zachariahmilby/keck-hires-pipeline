from pathlib import Path


class _Files:
    """
    Object to hold FITS file paths and ensure that a directory exists.
    """
    def __init__(self, directory: Path, file_type: str = 'fits'):
        self._directory = self.check_if_directory_exists(directory)
        self._file_type = file_type

    @staticmethod
    def check_if_directory_exists(directory: Path):
        if not directory.absolute().exists():
            raise OSError('Provided directory does not exist.')
        else:
            return directory.absolute()

    @property
    def paths(self) -> list[Path]:
        return sorted(self._directory.glob(f'*.{self._file_type}*'))


def make_directory(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True)
    return directory
