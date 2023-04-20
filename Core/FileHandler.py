import importlib.util
import json
import mne
import fnmatch
import os
import logging

log = logging.getLogger(__name__)


class FileHandler:
    def __init__(self, root_dir):
        """
        Initialize the FileHandler object with the root directory of the data.

        Parameters:
        root_dir (str): The root directory where the data is stored.
        """
        self.root_dir = root_dir
        self.data_dir = None

    def find(self, pattern):
        """
        Find files in the directory tree that match a given pattern.

        Parameters:
        pattern (str): The pattern to match the filenames with.

        Returns:
        found_files (list): A list of filepaths that match the given pattern.
        """
        found_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    found_files.append(os.path.join(root, name))
        if len(found_files) == 1:
            log.debug("Found one file")
            return found_files[0]
        else:
            log.debug("Found more than one file")
            return found_files

    def load_preproc_params(self, pattern="*.py"):
        """
        Load preprocessing parameters from a Python file.

        Parameters:
        pattern (str): The pattern to match the filename with.

        Returns:
        config (module): A Python module containing the preprocessing parameters.
        """
        config_fp = self.find(pattern=pattern)
        if not config_fp.__len__():
            log.warning("No config file found!")
        spec = importlib.util.spec_from_file_location("preprocessing_parameter_configuration", config_fp)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        log.info("Successfully loaded preprocessing params")
        return config

    def load_mapping(self, pattern="*.json"):
        """
        Load a JSON mapping file.

        Parameters:
        pattern (str): The pattern to match the filename with.

        Returns:
        mapping (dict): A dictionary containing the mapping information.
        """
        mapping_fp = self.find(pattern=pattern)
        if not mapping_fp.__len__():
            log.warning("No mapping file found!")
        with open(mapping_fp) as f:
            mapping = json.load(f)
        log.info("Successfully loaded mapping")
        return mapping

    def load_montage(self, pattern="*.bvef"):
        """
        Load a custom montage from a BrainVision EEG Format file.

        Parameters:
        pattern (str): The pattern to match the filename with.

        Returns:
        montage (mne.channels.Montage): An MNE montage object.
        """
        montage_fp = self.find(pattern=pattern)
        if not montage_fp.__len__():
            log.warning("No montage file found!")
        montage = mne.channels.read_custom_montage(fname=montage_fp)
        log.info("Successfully loaded montage")
        return montage

    def load_ica_reference(self, pattern="*ica.fif"):
        """
        Load an ICA reference file.

        Parameters:
        pattern (str): The pattern to match the filename with.

        Returns:
        ica_ref (mne.preprocessing.ICA): An MNE ICA object.
        """
        ica_ref_fp = self.find(pattern=pattern)
        if not ica_ref_fp.__len__():
            log.warning("No ica file found!")
        ica_ref = mne.preprocessing.read_ica(ica_ref_fp)
        log.info("Successfully loaded ica reference")
        return ica_ref


if __name__ == "__main__":
    fp = "D:\\EEG\\example"
    fl = FileHandler(root_dir=fp)
    fl.load_ica_reference()
