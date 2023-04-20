import os
import pathlib
import mne
import numpy as np
import logging


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

    # Make a copy of the epochs data to avoid modifying the original data
    epochs_tmp = epochs.copy()
    # Remove the last epoch if the number of epochs is odd to ensure even number of epochs for averaging
    n_epochs = epochs_tmp.get_data().shape[0]
    if not n_epochs % 2 == 0:
        epochs_tmp = epochs_tmp[:-1]
    # Get the number of epochs after removing the last epoch
    n_epochs = epochs_tmp.get_data().shape[0]
    # Invert every other epoch to create noise epochs
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    # Calculate the average of the inverted epochs to get the noise epochs
    noises = epochs_tmp.average().get_data()
    # Shuffle the noise epochs along the time axis to create shuffled noise epochs
    shuffled_noises = _shuffle_along_axis(noises, axis=1)
    # Calculate the signal epochs by subtracting the shuffled noise epochs from the original epochs
    signals = shuffled_noises.copy()
    for idx, noise in enumerate(shuffled_noises):
        for epoch in epochs.average().get_data():
            signal = epoch - noise
        signals[idx] = signal
    # Calculate the SNR for each epoch
    snr = signals[signals != 0] / noises[noises != 0]
    # Calculate the root-mean-square (RMS) of the SNR values
    rms = np.mean(np.sqrt(snr ** 2))

    return rms


def set_logger(logger, level):
    logger.setLevel(level.upper())
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level.upper())
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)