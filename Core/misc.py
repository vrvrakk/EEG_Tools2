import mne
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _shuffle_along_axis(data, axis):
    """
    Shuffle the values of a multi-dimensional array along a specified axis.

    Parameters
    ----------
    data : numpy.ndarray
        The multi-dimensional array to be shuffled.
    axis : int
        The axis along which the values should be shuffled.

    Returns
    -------
    numpy.ndarray
        The shuffled multi-dimensional array.
    """
    idx = np.random.rand(*data.shape).argsort(axis=axis)
    return np.take_along_axis(data, idx, axis=axis)


def snr(epochs):
    """
    Calculate signal-to-noise ratio (SNR) from EEG epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing the EEG epochs.

    Returns
    -------
    float
        SNR estimate based on root-mean-square (RMS) of the SNR values.

    """

    logger.info("Started SNR calculation.")

    # Make a copy of the epochs data to avoid modifying the original data
    epochs_tmp = epochs.copy()

    # Remove the last epoch if the number of epochs is odd to ensure even number of epochs for averaging
    n_epochs = epochs_tmp.get_data().shape[0]
    if not n_epochs % 2 == 0:
        epochs_tmp = epochs_tmp[:-1]
    n_epochs = epochs_tmp.get_data().shape[0]

    logger.debug(f"Number of epochs after removing the last epoch: {n_epochs}")

    # Invert every other epoch to create noise epochs
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]

    logger.debug("Inverted every other epoch to create noise epochs.")

    # Calculate the average of the inverted epochs to get the noise epochs
    noises = epochs_tmp.average().get_data()

    logger.debug("Calculated the average of the inverted epochs to get the noise epochs.")

    # Shuffle the noise epochs along the time axis to create shuffled noise epochs
    shuffled_noises = _shuffle_along_axis(noises, axis=1)

    logger.debug("Shuffled the noise epochs along the time axis to create shuffled noise epochs.")

    # Calculate the signal epochs by subtracting the shuffled noise epochs from the original epochs
    signals = shuffled_noises.copy()
    for idx, noise in enumerate(shuffled_noises):
        for epoch in epochs.average().get_data():
            signal = epoch - noise
        signals[idx] = signal

    logger.debug("Calculated the signal epochs by subtracting the shuffled noise epochs from the original epochs.")

    # Calculate the SNR for each epoch
    snr = signals[signals != 0] / noises[noises != 0]

    logger.debug("Calculated the SNR for each epoch.")

    # Calculate the root-mean-square (RMS) of the SNR values
    rms = np.mean(np.sqrt(snr ** 2))

    logger.info("Finished SNR calculation.")

    return rms


def set_logger(logger, level):
    """Configure a logger with a stream handler and a specified log level.

    Args:
        logger (logging.Logger): The logger instance to be configured.
        level (str): The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING').

    Returns:
        None

    """
    # Set the logging level of the logger instance to the specified level
    logger.setLevel(level.upper())

    # Create a StreamHandler to send log messages to the console
    ch = logging.StreamHandler()

    # Set the logging level of the StreamHandler to the specified level
    ch.setLevel(level.upper())

    # Create a Formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the Formatter to the StreamHandler
    ch.setFormatter(formatter)

    # Add the StreamHandler to the logger instance
    logger.addHandler(ch)

