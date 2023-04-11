from eeg_tools import utils

class Pipeline:
    def __init__(self, root_dir=None):
        # Directories
        root_dir = None
        raw_dir = None
        epoch_dir = None
        evoked_dir = None

        # Objects
        subject_id = list  # length in list defines how many subjects are being processed
        raws = None
        epochs = None
        evokeds = None

        # Object states - True if the objects have been processed
        processed_raws = False
        processed_epochs = False

        # Necessary files for preprocessing
        if root_dir:
            mapping = None
