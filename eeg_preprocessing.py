from EEG_Tools2.core.Pipeline import EEGPipeline
from EEG_Tools2.core.misc import *


root_dir = "C:/Users/vrvra/EEG_data"
pipeline = EEGPipeline(root_dir)

subject = pipeline.subjects[0]
# step 1
pipeline.concatenate_brainvision(subject=subject, ref_to_add="FCz", add_reference_channels=True)

filtering_params = pipeline.params.filtering
# step 2
pipeline.filtering(highpass=filtering_params['highpass'] , lowpass=filtering_params['lowpass'])

# saving :pipeline.save(data=pipeline.raw, subject=subject)
epoch_params = pipeline.params.epochs
pipeline.make_epochs(event_id=epoch_params['event_id'], exclude_event_id=None)
# save epochs
pipeline.save(data=pipeline.epochs, subject=subject, overwrite=True)


# AT END OF ALGORITHM: save snr
pipeline.snr_to_txt(subject=subject)