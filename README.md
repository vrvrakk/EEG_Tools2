# **EEG_Tools2**

This repository contains modules to preprocess and analyze electro-physiological brain data (mainly EEG). It is meant for data handling collected with [BrainVision](https://brainvision.com/) products.
## Prerequisites

Before installation, be sure to create an environment for adequate functionality. You can use the environment.yml file to create your own [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) environment.

## Installation

You'll need to clone the GitHub repository to your local machine. To do this, navigate to the repository on GitHub and click on the "Clone or download" button. You can then copy the HTTPS or SSH URL for the repository.

Next, open a terminal window on your local machine and use the following command to clone the repository:

`
git clone https://github.com/mxschlz/EEG_Tools2.git
`

## Main Ideas and Structure

The whole module serves as a convenience wrapper for the [MNE-Python](https://mne.tools/stable/index.html) library.
Here, instead of writing extended scripts for basic preprocessing, you can simply set all the parameters available for preprocessing with MNE-Python in a single `config.py` file.
The core modules can be found in the **core** directory. Here, the EEGPipeline is the main module for preprocessing EEG data. The `misc.py` module contains utility functions such as for calculating SNR ratios or setting the logging level for information output.
With the Analyzer you can load up your preprocessed evoked data for further analysis with MNE-Python. 
Be sure to include the setting and data files in the same root directory. A pipeline processing example can be found in `examples/run_pipeline.py`.
