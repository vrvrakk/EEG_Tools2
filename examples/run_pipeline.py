from core.misc import *
from core.Pipeline import EEGPipeline
import core

set_logger(logger=core.Pipeline.logger, level="debug")

# Instantiate
fp = "D:\\EEG\\example"
pl = EEGPipeline(root_dir=fp)

# run automated pipeline for one subject
pl.run(subjects=pl.subjects)