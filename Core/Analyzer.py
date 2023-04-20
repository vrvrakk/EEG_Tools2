from core.Pipeline import EEGPipeline
import mne
import os
import logging

log = logging.getLogger(__name__)


class EEGAnalyzer(EEGPipeline):

    def __init__(self, root_dir):
        super().__init__(root_dir)

    def get_evokeds_all_subjects(self, return_average=False, condition=None):
        all_evokeds = dict()
        for subject in self.subjects:
            path = os.path.join(f"{self.data_dir}", subject, f"{subject}-ave.fif")
            evokeds = mne.read_evokeds(path, condition=condition)
            for cond in evokeds:
                if cond.comment not in all_evokeds.keys():
                    all_evokeds[cond.comment] = [cond]
                else:
                    all_evokeds[cond.comment].append(cond)
        if return_average:
            evokeds_avrgd = dict()
            for key in all_evokeds:
                evokeds_avrgd[key] = mne.grand_average(all_evokeds[key])
            return all_evokeds, evokeds_avrgd
        else:
            return all_evokeds


if __name__ == "__main__":
    fp = "D:\\EEG\\example"
    analyzer = EEGAnalyzer(fp)
    evkds = analyzer.get_evokeds_all_subjects()
