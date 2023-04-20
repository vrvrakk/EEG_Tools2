import importlib.util
import json
import mne
import fnmatch
import os
import logging

log = logging.getLogger(__name__)


class FileHandler:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_dir = None

    def find(self, pattern):
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
        config_fp = self.find(pattern=pattern)
        if not config_fp.__len__():
            log.warning("No config file found!")
        spec = importlib.util.spec_from_file_location("preprocessing_parameter_configuration", config_fp)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        log.info("Successfully loaded preprocessing params")
        return config

    def load_mapping(self, pattern="*.json"):
        mapping_fp = self.find(pattern=pattern)
        if not mapping_fp.__len__():
            log.warning("No mapping file found!")
        with open(mapping_fp) as f:
            mapping = json.load(f)
        log.info("Successfully loaded mapping")
        return mapping

    def load_montage(self, pattern="*.bvef"):
        montage_fp = self.find(pattern=pattern)
        if not montage_fp.__len__():
            log.warning("No montage file found!")
        montage = mne.channels.read_custom_montage(fname=montage_fp)
        log.info("Successfully loaded montage")
        return montage

    def load_ica_reference(self, pattern="*ica.fif"):
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
