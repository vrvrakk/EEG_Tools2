from core.Pipeline import EEGPipeline  # import EEGPipeline class from core.Pipeline module
import mne  # import MNE library
import os  # import os module for operating system dependent functionality
import logging  # import logging module for logging purposes

log = logging.getLogger(__name__)  # create a logger object with the name of the current module


class EEGAnalyzer(EEGPipeline):
    """
    EEGAnalyzer is a class that extends the EEGPipeline class and provides additional functionality to analyze
    evoked data for all subjects.

    Args:
    - root_dir (str): the root directory of the data

    Attributes:
    - subjects (list): a list of all subject names

    Methods:
    - get_evokeds_all_subjects(return_average=False, condition=None): returns all evoked data for all subjects
    """

    def __init__(self, root_dir):
        """
        Initializes the EEGAnalyzer object.

        Args:
        - root_dir (str): the root directory of the data
        """
        super().__init__(root_dir)

    def get_evokeds_all_subjects(self, return_average=False, condition=None):
        """
        Returns all evoked data for all subjects.

        Args:
        - return_average (bool): whether to return the grand average of evoked data for each condition
        - condition (str): the condition to select (if None, returns all conditions)

        Returns:
        - all_evokeds (dict): a dictionary containing all evoked data for each subject and condition
        - evokeds_avrgd (dict): a dictionary containing the grand average of evoked data for each condition
        """
        all_evokeds = dict()  # initialize an empty dictionary to store evoked data for each subject and condition
        for subject in self.subjects:
            # get the path of the evoked data file for the current subject and condition
            path = os.path.join(f"{self.data_dir}", subject, f"{subject}-ave.fif")
            evokeds = mne.read_evokeds(path, condition=condition)  # read the evoked data from the file
            for cond in evokeds:
                if cond.comment not in all_evokeds.keys():  # if the condition is not already in the dictionary
                    all_evokeds[cond.comment] = [cond]  # add a new key-value pair to the dictionary
                else:  # if the condition is already in the dictionary
                    all_evokeds[cond.comment].append(cond)  # append the new evoked data to the existing list of evoked data for that condition
        if return_average:
            evokeds_avrgd = dict()
            for key in all_evokeds:
                evokeds_avrgd[key] = mne.grand_average(all_evokeds[key])
            return all_evokeds, evokeds_avrgd
        else:
            return all_evokeds


if __name__ == "__main__":
    fp = "D:\\EEG\\example"  # set the root directory of the data
    analyzer = EEGAnalyzer(fp)  # create an EEGAnalyzer object with the root directory
    evkds = analyzer.get_evokeds_all_subjects()  # get all evoked data for all subjects
