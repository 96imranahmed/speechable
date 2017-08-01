import sys
import logging
import numpy as np
import scipy
import scipy.signal
from python_speech_features.sigproc import preemphasis

def stft(x, fs, framesz, hop, two_sided=True, fft_size=None):
    '''
    Short Time Fourier Transform (STFT) - Spectral decomposition

    Input:
        x - signal (1-d array, which is amp/sample)
        fs - sampling frequency (in Hz)
        framesz - frame size (in seconds)
        hop - skip length (in seconds)
        two_sided - return full spectrogram if True
            or just positive frequencies if False
        fft_size - number of DFT points

    Output:
        X = 2d array time-frequency repr of x, time x frequency
    '''

    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    overlap_samp = framesamp - hopsamp

    _, _, X = scipy.signal.stft(x, fs, window='hann', nperseg=framesamp,
        noverlap=overlap_samp, nfft=fft_size, return_onesided=not two_sided)
    return X.T


def istft(X, fs, recon_size, hop, two_sided=True, fft_size=None):
    ''' Inverse Short Time Fourier Transform (iSTFT) - Spectral reconstruction

    Input:
        X - set of 1D time-windowed spectra, time x frequency
        fs - sampling frequency (in Hz)
        recon_size - Not used
        hop - skip rate between successive windows
        fft_size - number of DFT points

    Output:
        x - a 1-D array holding reconstructed time-domain audio signal
    '''
    if two_sided:
        framesamp = X.shape[1]
    else:
        framesamp = 2*(X.shape[1] - 1)
    hopsamp = int(hop*fs)
    overlap_samp = framesamp - hopsamp

    _, x = scipy.signal.istft(X.T, fs=fs, window='hann', nperseg=framesamp,
        nfft=fft_size, noverlap = overlap_samp,
        input_onesided=not two_sided)
    if recon_size is not None and recon_size != x.shape[0]:
        logger = logging.getLogger(__name__)
        logger.warn("Size of reconstruction ({}) does not match value of "
        "deprecated recon_size parameter ({}).".format(x.shape[0], recon_size))
    return x

def scale_spectrogram(spectrogram):
    mag_spec = np.abs(spectrogram)
    phases = np.unwrap(np.angle(spectrogram))

    mag_spec = np.sqrt(mag_spec)
    M = mag_spec.max()
    m = mag_spec.min()

    return (mag_spec - m)/(M - m), phases

def preprocess_signal(signal, sample_rate):
    """
    Preprocess a signal for input into a model

    Inputs:
        signal: Numpy 1D array containing waveform to process
        sample_rate: Sampling rate of the input signal

    Returns:
        spectrogram: STFT of the signal after resampling to 10kHz and adding
                 preemphasis.
        X_in: Scaled STFT input feature for the model
    """

    # Compute the spectrogram of the signal
    spectrogram = make_stft_features(signal, sample_rate)

    # Get the magnitude spectrogram
    mag_spec = np.abs(spectrogram)

    # Scale the magnitude spectrogram with a square root squashing, and percent
    # normalization
    X_in = np.sqrt(mag_spec)
    m = X_in.min()
    M = X_in.max()
    X_in = (X_in - m)/(M - m)

    return spectrogram, X_in

def process_signal(signal, sample_rate, model):
    """
    Compute the spectrogram and T-F embedding vectors for a signal using the
    specified model.

    Inputs:
        signal: Numpy 1D array containing waveform to process
        sample_rate: Sampling rate of the input signal
        model: Instance of model to use to separate the signal

    Returns:
        spectrogram: Numpy array of shape (Timeslices, Frequency) containing
                     the complex spectrogram of the input signal.
        vectors: Numpy array of shape (Timeslices, Frequency, Embedding)
    """

    # Preprocess the signal into an input feature
    spectrogram, X_in = preprocess_signal(signal, sample_rate)

    # Reshape the input feature into the shape the model expects and compute
    # the embedding vectors
    X_in = np.reshape(X_in, (1, X_in.shape[0], X_in.shape[1]))
    vectors = model.get_vectors(X_in)

    return spectrogram, vectors

def featurize_spectrogram(spectrogram):
    """
    Takes in a spectrogram and outputs a normalized version for consumption by
    the model
    """
    # Get the magnitude spectrogram and phases
    X_input = np.abs(spectrogram)
    phases = np.unwrap(np.angle(spectrogram))

    # Normalize the magnitude spectrogram
    X_input = np.sqrt(X_input)
    X_max = X_input.max()
    X_min = X_input.min()
    X_input = (X_input - X_min)/(X_max - X_min)

    return X_input, phases, X_max, X_min

def undo_preemphasis(preemphasized_signal,coeff=0.95):
    """
    Function to undo the preemphasis of an input signal. The preemphasised
    signal p is computed from the signal s by the relation
                    p(n) = s(n) - coeff*s(n-1)
    with p(0) = s(0).  The inverse operation constructs the signal from the
    preemphasized signal with the recursion relation
                    s(n) = p(n) + coeff*s(n-1)
    Inputs:
        preemphasized_signal:  numpy array containing preemphasised signal
        coeff:   coefficient used to compute the preemphasized signal
    Returns:
        signal: numpy array containing the signal without preemphasis
    """

    # Get the length of the input and preallocate the output array
    length = preemphasized_signal.shape[0]
    signal = np.zeros(length)

    # Set the initial element of the signal
    signal[0] = preemphasized_signal[0]

    # Use the recursion relation to compute the output signal
    for i in range(1,length):
        signal[i] = preemphasized_signal[i] + coeff*signal[i-1]

    return signal

def make_stft_features(signal, sample_rate,
                       output_sample_rate=1e4,
                       window_size=0.0512, overlap=0.0256,
                       preemphasis_coeff=0.95, fft_size=512):
    '''
    Function to take in a signal, resample it to output_sample_rate,
    normalize it, and compute the magnitude spectrogram.
    Inputs:
        signal: 1D numpy array containing signal to featurize (np.ndarray)
        sample_rate: sampling rate of signal (int)
        output_sample_rate: sample rate of signal after resampling (int)
        window_size: length of stft window in seconds (float)
        overlap: amount of overlap for stft windows (float)
        preemphasis: preemphasis coefficient (float)
        fft_size: length (in seconds) of DFT window (float)

    Returns:
        spectrogram: 2D numpy array with (Time, Frequency) components of
                     input signals (np.ndarray)
    '''

    # Downsample the signal to output_sample_rate
    resampled = scipy.signal.resample_poly(signal,100,
                              int(sample_rate/output_sample_rate*100))

    # Do preemphasis on the resampled signal
    preemphasised = preemphasis(resampled,preemphasis_coeff)

    # Normalize the downsampled signal
    normalized = (preemphasised - preemphasised.mean())/preemphasised.std()

    # Get the spectrogram
    spectrogram = stft(normalized,output_sample_rate,
                       window_size,overlap,two_sided=False, fft_size=fft_size)

    return spectrogram

def low_pass_filt(signal, cutoff, fs, order=5):
    """
    Function to take in a signal, and low-pass filter it with a cutoff
    """
    wn = cutoff/fs
    B, A = scipy.signal.butter(order, wn, output ='ba')
    signal = scipy.signal.filtfilt(B,A, signal, axis = 0) #To avoid problems with two channel signals etc.
    return signal

def verify_COLA(frame_size, overlap_size, fs):
    window = 'hann' 
    framesamp = int(frame_size*fs) #frame size (in seconds) nperseg
    hopsamp = int(overlap_size*fs) #overlap (in seconds)
    overlap_samp = framesamp - hopsamp #noverlap
    if not scipy.signal.check_COLA(window, framesamp, overlap_samp):
        print('Non COLA compliant input')
        return False
    else:
        return True