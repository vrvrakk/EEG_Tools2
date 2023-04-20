from core.misc import *
from core.FileHandler import FileHandler
from core.FigureHandler import FigureHandler
import os
import mne
import logging
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
import numpy as np
import matplotlib
matplotlib.use("qt5agg")

logger = logging.getLogger(__name__)

# TODO: implement FigureHandler
# TODO: subject handling sucks --> especially bad when saving data --> SubjectHandler?


class EEGPipeline(FileHandler):
    def __init__(self, root_dir):  # must be given
        """
        self.root_dir is the root directory. This is where data and config files must be located.
        To guarantee full pipeline functionality, folder Structure MUST be as follows:
        root_dir
            |- setting
                |- mapping.json
                |- ica_reference.fif  # optional
                |- config.py
                |- montage.bvef
            |- data
                |- sub_01
                |- sub_02
                |- sub_03
                ...
                |- sub_n
        """
        log.info("Instantiating EEG pipeline ... ")
        super().__init__(root_dir)
        self.root_dir = root_dir  # must direct to raw eeg data directories
        self.filehandler = FileHandler(self.root_dir)
        self.params = self.filehandler.load_preproc_params()  # load preprocessing parameters
        self.mapping = self.filehandler.load_mapping()  # load electrode mapping
        self.montage = self.filehandler.load_montage()  # load custom montage
        if self.params.ica["rejection_mode"] == "auto":
            self.ica_ref = self.filehandler.load_ica_reference()  # load ica reference if mode is automatic

        # Directories for data and figures
        self.data_dir = os.path.join(self.root_dir, "data")  # data directory
        self.figure_dir = None  # figure directory

        # Infer subjects from data directory
        self.subjects = os.listdir(self.data_dir)  # infer subjects from data directory
        self.subject_currently_processing = None  # placeholder for single subject

        # Objects for single subject
        self.brainvision_files = list  # here lie all the brainvision files
        self.raw = mne.io.Raw
        self.events = np.ndarray
        self.epochs = mne.epochs.Epochs
        self.evokeds = mne.evoked.EvokedArray
        self.ICA = mne.preprocessing.ICA
        self.ar = AutoReject

        # State dictionary to keep track of the processing state
        self.state = dict(concatenated_brainvision=False,
                          filtered=False,
                          epoched=False,
                          rereferenced=False,
                          applied_ica=False,
                          rejected_epochs=False,
                          averaged_epochs=False)

        # Keep track of SNR
        self.SNR_trend = dict()

    def concatenate_brainvision(self, subject, preload=True, add_reference_channels=True, ref_to_add=None, save=False):
        """
        Concatenates BrainVision files for a given subject.

        Parameters:
            - subject (str): The subject identifier to specify the files to be concatenated.
            - preload (bool): If True, preload the data into memory. Defaults to True.
            - add_reference_channels (bool): If True, add reference channels. Defaults to True.
            - ref_to_add (list or None): List of reference channels to be added. If None, an error will be raised if
                                         add_reference_channels is set to True. Defaults to None.
            - save : bool
                If True, save data. Defaults to False.

        Returns:
            None

        Modifies:
            - self.raw: Updates the self.raw object with concatenated raw data.
            - self.state: Updates the state with 'concatenated_brainvision' set to True.

        Raises:
            - ValueError: If ref_to_add is None and add_reference_channels is set to True.

        """
        log.info(f"Concatenating BrainVision files for {subject}...")
        # Find all vhdr files for the given subject
        raw_files = list()
        brainvision_path = self.filehandler.find(pattern="*vhdr")
        self.brainvision_files = [vhdr for vhdr in brainvision_path if subject in vhdr]
        self.subject_currently_processing = subject
        log.debug(f"Found {len(self.brainvision_files)} BrainVision files for {subject}")
        # Read and concatenate BrainVision files
        for file in self.brainvision_files:
            log.debug(f"Reading {file}...")
            raw_files.append(mne.io.read_raw_brainvision(vhdr_fname=file, preload=preload))  # read BrainVision files.
        self.raw = mne.concatenate_raws(raw_files)  # make raw files
        log.info(f"Successfully concatenated raw file for {subject}")
        # Rename channels if mapping is provided
        if self.mapping:
            log.info("Renaming channels...")
            self.raw.rename_channels(self.mapping)
        # Add reference channels if requested
        if add_reference_channels:
            log.info("Adding reference channels...")
            if ref_to_add is not None:
                self.raw.add_reference_channels(ref_to_add)
            else:
                raise ValueError("Need to add reference channel")
        # Set montage if provided
        if self.montage:
            log.info("Setting montage...")
            self.raw.set_montage(self.montage)
        # Update state
        self.state["concatenated_brainvision"] = True
        if save:
            log.info("Saving data...")
            self.save(data=self.raw, subject=self.subject_currently_processing)

    def filtering(self, highpass=None, lowpass=None, notch=None, picks=None,
                  filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                  n_jobs=None, method='fir', iir_params=None, phase='zero',
                  fir_window='hamming', fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'),
                  pad='reflect_limited', verbose=None, save=False):
        """
        Applies filtering to the raw data.

        Parameters:
            - highpass (float or None): The frequency (in Hz) for the highpass filter. If None, no highpass filter is applied.
            - lowpass (float or None): The frequency (in Hz) for the lowpass filter. If None, no lowpass filter is applied.
            - notch (float or None): The frequency (in Hz) for the notch filter. If None, no notch filter is applied.
            - picks (list, slice, or None): Channels to be included for filtering. If None, all channels are included.
            - filter_length (str, int, or None): Length of the FIR filter to be applied. If 'auto', it is calculated based on the
                                                sampling rate. If int, it specifies the length of the filter in samples. Defaults to 'auto'.
            - l_trans_bandwidth (float, str, or None): Width of the transition band for the lowpass filter. If 'auto', it is calculated
                                                        based on the filter length. Defaults to 'auto'.
            - h_trans_bandwidth (float, str, or None): Width of the transition band for the highpass filter. If 'auto', it is calculated
                                                        based on the filter length. Defaults to 'auto'.
            - n_jobs (int or None): Number of parallel jobs to run for FIR filtering. If None, it uses the default number of jobs
                                    determined by MNE-Python. Defaults to None.
            - method (str): Filter design method. Supports 'fir' and 'iir'. Defaults to 'fir'.
            - iir_params (dict or None): Parameters for IIR filtering. If None, default parameters are used.
            - phase (str): Phase of the filter. Supports 'zero', 'minimum', and 'linear'. Defaults to 'zero'.
            - fir_window (str or None): Window to use for FIR filter design. Supports 'hamming', 'hann', 'blackman', 'bartlett',
                                        'flattop', 'tukey', None. Defaults to 'hamming'.
            - fir_design (str): Method to use for FIR filter design. Supports 'firwin' and 'firwin2'. Defaults to 'firwin'.
            - skip_by_annotation (tuple): Annotations to skip during filtering. Defaults to ('edge', 'bad_acq_skip').
            - pad (str or None): Padding method to use for FIR filtering. Supports 'reflect_limited' and None. Defaults to 'reflect_limited'.
            - verbose (bool, str, int, or None): Verbosity level of the filter function. Defaults to None.
            - save (bool) : If True, save data. Defaults to False.

        Returns:
            None

        Modifies:
            - self.raw: Updates the self.raw object with filtered data.
            - self.state: Updates the state with 'filtered' set to True.

        """
        if not isinstance(self.raw, mne.io.brainvision.brainvision.RawBrainVision):
            log.error("Raw instance does not follow MNE convention!")
        else:
            log.info("Raw instance follows MNE convention.")

        if highpass or lowpass:
            log.info(f"Applying {'highpass' if highpass else 'lowpass'} filter with {method} method...")
            log.debug(
                f"Filter parameters: l_freq={highpass}, h_freq={lowpass}, picks={picks}, filter_length={filter_length}, l_trans_bandwidth={l_trans_bandwidth}, h_trans_bandwidth={h_trans_bandwidth}, method={method}, iir_params={iir_params}, phase={phase}, fir_window={fir_window}, fir_design={fir_design}, n_jobs={n_jobs}, skip_by_annotation={skip_by_annotation}, pad={pad}, verbose={verbose}")

            self.raw.filter(l_freq=highpass, h_freq=lowpass,
                            picks=picks, filter_length=filter_length, l_trans_bandwidth=l_trans_bandwidth,
                            h_trans_bandwidth=h_trans_bandwidth, method=method, iir_params=iir_params, phase=phase,
                            fir_window=fir_window, fir_design=fir_design, n_jobs=n_jobs,
                            skip_by_annotation=skip_by_annotation, pad=pad, verbose=verbose)
            log.info(f"{'Highpass' if highpass else 'Lowpass'} filter applied successfully!")

        if notch:
            log.info(f"Applying notch filter with {notch}Hz center frequency...")
            self.raw.notch_filter(notch)
            log.info("Notch filter applied successfully!")

        self.state["filtered"] = True
        if save:
            log.info("Saving filtered data...")
            self.save(data=self.raw, subject=self.subject_currently_processing)
            log.info("Filtered data saved successfully!")

    def make_epochs(self, exclude_event_id, event_id, preload=True, tmin=-0.2, tmax=0.5, baseline=(None, 0),
                    picks=None, reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None,
                    detrend=None, on_missing="raise", reject_by_annotation=True, metadata=None, event_repeated="error",
                    verbose=None, save=False):
        """
        Cut time series data into epochs.

        Parameters:
        ----------
        exclude_event_id : int or list of int | None
            The event id(s) to exclude from epoching. If None, no events will be excluded.
        event_id : dict | None
            The event id dictionary. The keys represent the event names and the values represent the event codes.
        preload : bool
            If True, preload the data into memory. Defaults to True.
        tmin : float
            The start time of the epoch in seconds relative to the event onset. Defaults to -0.2.
        tmax : float
            The end time of the epoch in seconds relative to the event onset. Defaults to 0.5.
        baseline : tuple | None
            The baseline period to use for baseline correction. The tuple contains two values: (start, stop) where start
            is the start time of the baseline period in seconds relative to the event onset, and stop is the end time of
            the baseline period in seconds relative to the event onset. If None, no baseline correction will be applied.
            Defaults to (None, 0).
        picks : array-like of int | None
            The indices of the channels to include in the epoching. If None, all channels will be included. Defaults to None.
        reject : dict | None
            The rejection parameters for each channel type. The dictionary keys represent the channel types and the values
            represent the rejection criteria. If None, no rejection will be applied. Defaults to None.
        flat : dict | None
            The flatness parameters for each channel type. The dictionary keys represent the channel types and the values
            represent the flatness criteria. If None, no flatness rejection will be applied. Defaults to None.
        proj : bool
            If True, apply SSP projection vectors. Defaults to True.
        decim : int
            The decimation factor to apply during epoching. Defaults to 1.
        reject_tmin : float | None
            The start time of the time window to use for rejection threshold calculation. If None, the value of tmin will
            be used. Defaults to None.
        reject_tmax : float | None
            The end time of the time window to use for rejection threshold calculation. If None, the value of tmax will
            be used. Defaults to None.
        detrend : int | None
            The type of detrending to apply to the data. If None, no detrending will be applied. Defaults to None.
        on_missing : str
            The action to take when an event is missing in the event_id dictionary. Options are 'raise', 'warn', or 'ignore'.
            Defaults to 'raise'.
        reject_by_annotation : bool
            If True, reject epochs based on annotations. Defaults to True.
        metadata : None
            Not used. Defaults to None.
        event_repeated : str
            The action to take when an event is repeated. Options are 'error', 'drop', or 'merge'. Defaults to 'error'.
        verbose : bool | None
            If True, print verbose output. If None, use the value of self.verbose
        save : bool
            If True, save data. Defaults to False.

        Returns:
        -------
        None

        Notes:
        ------
        This method cuts the time series data into epochs based on the provided event ids and time window parameters. It
        applies optional preprocessing steps such as baseline correction, rejection of epochs based on rejection criteria
        and flatness criteria, and SSP projection if specified. The resulting epochs are stored in the 'epochs' attribute
        of the object, and the state of the object is updated to indicate that the data has been epoched.

        """

        log.info("Cutting time series data into epochs ... ")
        self.events = mne.pick_events(events=mne.events_from_annotations(self.raw)[0],
                                      exclude=exclude_event_id)
        log.info(f"Number of events after excluding unwanted events: {len(self.events)}")
        self.epochs = mne.Epochs(self.raw, events=self.events, event_id=event_id, preload=preload, tmin=tmin, tmax=tmax,
                                 baseline=baseline,
                                 picks=picks, reject=reject, flat=flat, proj=proj, decim=decim, reject_tmin=reject_tmin,
                                 reject_tmax=reject_tmax, detrend=detrend, on_missing=on_missing,
                                 reject_by_annotation=reject_by_annotation, metadata=metadata,
                                 event_repeated=event_repeated, verbose=verbose)
        log.info(f"Number of epochs: {len(self.epochs)}")
        self.SNR_trend["initial"] = snr(self.epochs)
        log.info(f"Initial signal-to-noise ratio: {self.SNR_trend['initial']}")
        self.state["epoched"] = True
        if save:
            self.save(data=self.epochs, subject=self.subject_currently_processing)

    def rereference(self, ref_type="average", ransac_params=None, reference_electrode=None, save=False):
        """
        Rereference the epochs data to a new reference type.

        Parameters:
        ----------
        ref_type: str, optional (default: "average")
            The type of reference to apply. Possible values are "average", "REST", "linked_mastoids", or "custom".
        ransac_params: dict, optional (default: None)
            Parameters to configure the Ransac algorithm used for bad channel detection when ref_type is "average".
            Should be a dictionary with keys "n_channels" and "distance_threshold". Only applicable when ref_type is "average".
        reference_electrode: str or list of str, optional (default: None)
            Electrode(s) to use as the reference when ref_type is "custom". Should be the name of the electrode(s) in the
            Montage or Info object. Only applicable when ref_type is "custom".
        save : bool
            If True, save data. Defaults to False.

        Returns:
        -------
        None

        Notes:
        ------
        This method applies rereferencing to the epochs data based on the specified reference type. The options for reference
        type are: "average", "REST", "linked_mastoids", or "custom". The resulting epochs data are stored in the 'epochs'
        attribute of the object, and the state of the object is updated to indicate that the data has been rereferenced.

        """

        log.info(f"Rereferencing data according to type {ref_type} ... ")
        if not self.epochs:
            log.error("No epochs found. Make sure to create epochs from continuous data before re-referencing!")
        if ref_type == "average":
            if ransac_params:
                ransac = Ransac(**ransac_params)
            else:
                ransac = Ransac()
            ransac.fit(self.epochs)
            self.epochs.average().plot(exclude=[])
            bads = input(
                "Enter bad sensors here (separate several bad sensors via spacebar): ").split()
            if len(bads) != 0 and bads not in ransac.bad_chs_:
                ransac.bad_chs_.extend(bads)
            self.epochs = ransac.transform(self.epochs)
            self.epochs.info['bads'] = ransac.bad_chs_
            self.epochs.set_eeg_reference(ref_channels=ref_type, projection=True)
            self.epochs.apply_proj()
            log.info("Data has been rereferenced using average reference.")
        elif ref_type == "REST":
            sphere = mne.make_sphere_model("auto", "auto", self.epochs.info)
            src = mne.setup_volume_source_space(
                sphere=sphere, exclude=30., pos=5.)
            forward = mne.make_forward_solution(
                self.epochs.info, trans=None, src=src, bem=sphere)
            self.epochs = self.epochs.set_eeg_reference("REST", forward=forward)
            log.info("Data has been rereferenced using REST reference.")
        elif ref_type == "linked_mastoids":
            self.epochs = self.epochs.set_eeg_reference(["TP9", "TP10"])
            log.info("Data has been rereferenced using linked mastoids reference.")
        elif ref_type == "custom":
            self.epochs = self.epochs.set_eeg_reference(reference_electrode)
            log.info("Data has been rereferenced using custom reference.")
        self.SNR_trend["after_reref"] = snr(self.epochs)
        self.state["rereferenced"] = True
        if save:
            self.save(data=self.epochs, subject=self.subject_currently_processing)

    def apply_ica(self, n_components=None, method="fastica",
                  threshold="auto", rejection_mode="manual", noise_cov=None, random_state=None, fit_params=None,
                  max_iter="auto", verbose=None, save=False):
        """
        Apply Independent Component Analysis (ICA) to the data.

        Parameters:
        ----------
        n_components : int | None
            The number of components to decompose the data into. If None, all components will be retained.
        method : str
            The ICA method to use. Default is "fastica".
        threshold : str | float
            The threshold to use for artifact detection. Default is "auto".
        rejection_mode : str
            The rejection mode for artifact detection. Can be "manual" or "auto". Default is "manual".
        noise_cov : instance of Covariance | None
            The noise covariance used for whitening the data. Default is None.
        random_state : int | None | np.random.RandomState
            The random state to use for reproducibility. Default is None.
        fit_params : dict | None
            Additional parameters to pass to the ICA estimator. Default is None.
        max_iter : int | str
            The maximum number of iterations to run for the ICA algorithm. Default is "auto".
        verbose : bool | None
            Whether to print verbose output. Default is None.
        save : bool
            If True, save data. Defaults to False.

        Returns:
        --------
        None
        """
        log.info("Applying independent-component-analysis ... ")
        self.ICA = ICA(n_components=n_components, method=method, noise_cov=noise_cov, random_state=random_state,
                       fit_params=fit_params, max_iter=max_iter, verbose=verbose)
        if rejection_mode == "manual":
            log.info("Performing manual ICA rejection ...")
            self.ICA.fit(self.epochs)
            self.ICA.plot_components(picks=None)
            self.ICA.plot_sources(self.epochs, start=0, stop=15, show_scrollbars=False, block=True)
            self.ICA.exclude = list(
                (input("Enter components to exclude here (separate several components via spacebar): ").split()))
            self.ICA.exclude = [int(x) for x in self.ICA.exclude]
            self.ICA.apply(self.epochs, exclude=self.ICA.exclude)
        if rejection_mode == "auto":
            log.info("Performing automatic ICA rejection ...")
            self.ICA.fit(self.epochs)
            ref = self.ica_ref  # reference ICA containing blink and saccade components.
            labels = list(ref.labels_.keys())  # .labels_ dict must contain "blinks" key with int values.
            components = list(ref.labels_.values())
            for component, label in zip(components, labels):
                mne.preprocessing.corrmap([ref, self.ICA], template=(0, component[0]),
                                          label=label, plot=False, threshold=threshold)
                self.ICA.apply(self.epochs, exclude=self.ICA.labels_["blinks"])  # apply ICA
        self.SNR_trend["after_ica"] = snr(self.epochs)
        self.state["applied_ica"] = True
        if save:
            self.save(data=self.epochs, subject=self.subject_currently_processing)

    def reject_epochs(self, mode="autoreject", reject=None, flat=None, n_interpolate=(1, 4, 32), consensus=None,
                      cv=10, thresh_method="bayesian_optimization", n_jobs=-1, random_state=None, picks=None,
                      verbose=None, save=False):
        """
        Reject epochs based on specified mode.

        Parameters
        ----------
        mode : str, optional
            Epoch rejection mode. Can be either "autoreject" or "threshold", by default "autoreject".
        reject : dict | None, optional
            Rejection parameters for rejecting bad epochs. Used only when mode is "threshold", by default None.
        flat : dict | None, optional
            Rejection parameters for rejecting flat epochs. Used only when mode is "threshold", by default None.
        n_interpolate : list of int, optional
            Number of channels to interpolate in each step, by default [1, 4, 32].
            Used only when mode is "autoreject".
        consensus : float | None, optional
            Consensus threshold for rejecting epochs, by default None.
            Used only when mode is "autoreject".
        cv : int, optional
            Number of cross-validation folds, by default 10.
            Used only when mode is "autoreject".
        thresh_method : str, optional
            Thresholding method, can be "bayesian_optimization" or "random_search", by default "bayesian_optimization".
            Used only when mode is "autoreject".
        n_jobs : int, optional
            Number of jobs to run in parallel, by default -1.
            Used only when mode is "autoreject".
        random_state : int | None, optional
            Seed used for random number generation, by default None.
            Used only when mode is "autoreject".
        picks : list of str | None, optional
            List of channel names to use for epoch rejection, by default None.
        verbose : bool | None, optional
            Verbosity level. If True, print status messages, by default None.
            Used only when mode is "autoreject".
        save : bool
            If True, save data. Defaults to False.

        Returns
        -------
        None
        """
        log.info(f"Rejecting epochs based on {mode} ... ")
        if mode == "autoreject":
            log.info("Using AutoReject for epoch rejection...")
            self.ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs, consensus=consensus, cv=cv,
                                 thresh_method=thresh_method, random_state=random_state, verbose=verbose, picks=picks)
            self.ar.fit(self.epochs)
            self.epochs, reject_log = self.ar.transform(self.epochs, return_log=True)
        elif mode == "threshold":
            log.info("Using threshold-based rejection for epoch rejection...")
            self.epochs = self.epochs.drop_bad(reject=reject, flat=flat)
        else:
            log.error("Right now, only mode -> autoreject or manual is implemented. Please set the mode to autoreject "
                      "or manual!")
        self.SNR_trend["after_rejection"] = snr(self.epochs)
        self.state["rejected_epochs"] = True
        if save:
            self.save(data=self.epochs, subject=self.subject_currently_processing)

    def average_epochs(self, baseline=None, by_event_type=False, save=False):
        """
        Average epochs by event type.

        Parameters
        ----------
        baseline : tuple | None, optional
            The time range to be used as the baseline correction, in seconds.
            Can be a tuple of (start, end) or None, by default None.
            If provided, baseline correction will be applied to the epochs before averaging.
        by_event_type : bool, optional
            If True, average epochs separately for each event type, by default False.
            If False, average epochs across all event types.
        save : bool
            If True, save data. Defaults to False.

        Returns
        -------
        None
        """
        log.info(f"Averaging epochs by event type -> {by_event_type} ... ")
        if baseline:
            log.info(f"Applying baseline correction with {baseline} ... ")
            self.epochs.apply_baseline(baseline)
            self.SNR_trend["after_baseline"] = snr(self.epochs)
        else:
            log.info("No baseline correction applied.")
        self.evokeds = self.epochs.average(by_event_type=by_event_type)
        self.state["averaged_epochs"] = True
        if save:
            self.save(data=self.evokeds, subject=self.subject_currently_processing)
        log.info("Epochs averaged successfully.")

    def snr_to_txt(self, subject):
        log.info(f"Saving SNR trend for {subject} as .txt file ... ")
        to_path = os.path.join(self.data_dir, subject, f"SNR_trend_{subject}.txt")
        with open(to_path, 'w') as f:
            for key, value in self.SNR_trend.items():
                f.write('%s:%s\n' % (key, value))
        log.info(f"SNR trend saved as .txt file for {subject}!")

    def save(self, data, subject, overwrite=True):
        """Save MNE-Python data to a file for a specified subject.

        Args:
            data: The MNE-Python data object to be saved (e.g., mne.io.brainvision.brainvision.RawBrainVision,
                  mne.epochs.Epochs, mne.evoked.Evoked, or list of mne.evoked.Evoked objects).
            subject (str): The subject identifier for the data being saved.
            overwrite (bool): Whether to overwrite an existing file with the same name (default is True).

        Returns:
            None

        """
        # Log that the file is being saved for the specified subject
        log.info(f"Saving file for subject {subject} ... ")

        # Determine the appropriate file extension based on the type of data being saved
        if isinstance(data, mne.io.brainvision.brainvision.RawBrainVision):
            ext = '_raw.fif'
        elif isinstance(data, mne.epochs.Epochs):
            ext = '-epo.fif'
        elif isinstance(data, mne.evoked.Evoked):
            ext = '-ave.fif'
        elif isinstance(data, list):
            ext = '-ave.fif'

        # Save the data to a file in the appropriate format and location
        if isinstance(data, list):
            log.info(f"Saving list of evoked data for subject {subject} ...")
            mne.write_evokeds(os.path.join(f"{self.data_dir}", subject, f"{subject}{ext}"), evoked=data,
                              overwrite=overwrite)
        else:
            log.info(f"Saving data for subject {subject} ...")
            data.save(os.path.join(f"{self.data_dir}", subject, f"{subject}{ext}"), overwrite=overwrite)

    def run(self, subjects, concatenate=True, filtering=True, epochs=True, rereference=True,
            ica=True, reject=True, averaging=True, snr_to_text=True):
        """
        Runs the pipeline for each subject in the given list of subjects.

        Parameters:
            subjects (list): A list of subject names.
            concatenate (bool): Whether or not to concatenate the data. Default is True.
            filtering (bool): Whether or not to filter the data. Default is True.
            epochs (bool): Whether or not to create epochs. Default is True.
            rereference (bool): Whether or not to re-reference the data. Default is True.
            ica (bool): Whether or not to apply Independent Component Analysis (ICA) to the data. Default is True.
            reject (bool): Whether or not to reject epochs. Default is True.
            averaging (bool): Whether or not to average the epochs. Default is True.
            snr_to_text (bool): Whether or not to save the Signal-to-Noise Ratio (SNR) of the epochs to a text file. Default is True.

        Returns:
            None
        """
        log.info("Starting pipeline ... ")
        if not self.subjects == list():
            log.error(f"{self}.subjects must be a list of subject names, not {type(self.subjects)}!")
        for subject in subjects:
            log.info(f"Processing subject {subject}")
            if concatenate:
                log.info(f"Concatenating data for subject {subject} ... ")
                self.concatenate_brainvision(subject=subject, **self.params.concatenate)
            if filtering:
                log.info(f"Filtering data for subject {subject} ... ")
                self.filtering(**self.params.filtering)
            if epochs:
                log.info(f"Epoching data for subject {subject} ... ")
                self.make_epochs(**self.params.epochs)
            if rereference:
                log.info(f"Rereferencing data for subject {subject} ... ")
                self.rereference(**self.params.rereference)
            if ica:
                log.info(f"Applying ICA for subject {subject} ... ")
                self.apply_ica(**self.params.ica)
            if reject:
                log.info(f"Rejecting epochs for subject {subject} ... ")
                self.reject_epochs(**self.params.reject)
            if averaging:
                log.info(f"Averaging epochs for subject {subject} ... ")
                self.average_epochs(**self.params.averaging)
            if snr_to_text:
                log.info(f"Saving SNR trend as .txt file for subject {subject} ... ")
                self.snr_to_txt(subject=self.subject_currently_processing)


if __name__ == "__main__":
    set_logger(logger=log, level="debug")

    # Instantiate
    fp = "D:\\EEG\\example"
    pl = EEGPipeline(root_dir=fp)

    # run automated pipeline for one subject
    pl.run(subjects=pl.subjects)
