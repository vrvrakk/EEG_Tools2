concatenate = dict(preload=True,  # must be True per default
                   add_reference_channels=False,
                   ref_to_add="FCz",  # must be set if add_reference_channel is True
                   save=True)  # save concatenated raw
filtering = dict(highpass=1.0,  # highpass filter frequency
                 lowpass=40.0,  # lowpass filter frequency
                 notch=None,  # float or array or float
                 picks=None,  # channels to include
                 filter_length='auto',  # length of FIR filter
                 l_trans_bandwidth='auto',  # Width of the transition band at the low cut-off frequency in Hz
                 h_trans_bandwidth='auto',  # Width of the transition band at the high cut-off frequency in Hz
                 n_jobs=None,  # Number of jobs to run in parallel
                 method='fir',  # filtering method --> FIR or IIR
                 iir_params=None,  # Dictionary of parameters to use for IIR filtering
                 phase='zero',  # Phase of the filter, only used if method='fir'
                 fir_window='hamming',  # The window to use in FIR design
                 fir_design='firwin',
                 skip_by_annotation=('edge', 'bad_acq_skip'),
                 pad='reflect_limited',  # The type of padding to use
                 verbose=True,  # set verbosity level
                 save=False)  # save filtered raw file
epochs = dict(event_id=dict(A=1,
                            B=2,
                            C=3,
                            D=4,
                            E=5),  # IDs of events
              tmin=-0.2,  # minumum time
              tmax=1.0,  # maximum time
              detrend=None,  # If 0 or 1, the data channels (MEG and EEG) will be detrended when loaded. 0 is a constant (DC) detrend, 1 is a linear detrend.
              baseline=None,  # sets the baseline
              exclude_event_id=None,  # event ID to be excluded
              preload=True,  # preloaded into memory storage
              reject=None,  # rejection criteria for high values  (eeg=100e-6)
              flat=None,  # rejection criteria for low values  (eeg=1e-6)
              proj=True,  # Apply SSP projection vectors
              decim=1,  # Factor by which to subsample the data.
              reject_tmin=None,  # Start of the time window used to reject epochs
              reject_tmax=None,  # End of the time window used to reject epochs
              on_missing="raise",  # What to do if one or several event ids are not found in the recording.
              reject_by_annotation=True,  # Whether to reject based on annotations
              metadata=None,  # A pandas.DataFrame specifying metadata about each epoch.
              event_repeated="error",  # How to handle duplicates in self.events[:, 0]
              verbose=True,  # verbosity level
              save=False)  # save epochs
rereference = dict(ref_type="average",  # type of referencing
                   ransac_params=dict(n_resample=50,
                                      min_channels=0.25,
                                      min_corr=0.75,
                                      unbroken_time=0.4,
                                      n_jobs=-1,
                                      verbose=True,
                                      picks=None),  # only used if ref_type == average
                   reference_electrode=None,  # reference electrode if type != average
                   save=False)  # save rereferenced epochs
ica = dict(n_components=0.95,  # Number of principal components
           method="fastica",  # The ICA method to use in the fit method
           rejection_mode="automatic",  # can be auto or manual
           threshold=0.75,  # correlation threshold if mode == auto
           noise_cov=None,  # Noise covariance used for pre-whitening
           random_state=None,  # sets the random seed to always ensure equal output
           fit_params=None,  # Additional parameters passed to the ICA estimator
           max_iter="auto",  # Maximum number of iterations during fit
           verbose=True,  # verbosity level
           save=False)  # save ica corrected epochs
reject = dict(mode="threshold",  # can be autoreject or threshold
              reject=dict(eeg=100e-6),  # only used if mode == "threshold"
              flat=dict(eeg=1e-6),  # only used if mode == "threshold"
              n_interpolate=[1, 4, 32],  # The values to try for the number of channels for which to interpolate
              cv=10,  # number of cross-validations
              consensus=None,  # The values to try for percentage of channels that must agree as a fraction of the total number of channels
              thresh_method="bayesian_optimization",
              n_jobs=-1,  # number of jobs to run in parallel
              random_state=None,  # sets the random seed
              picks=None,  # channels to include
              verbose=True,  # verbosity level
              save=True)  # save final epochs
averaging = dict(baseline=(None, 0),  # sets a baseline
                 by_event_type=True,  # if True: outputs an evoked list sorted by events
                 save=True)  # save evokeds
