from scipy.io import wavfile
import numpy as np
import math
import pyaudio
from matplotlib import pyplot as plt

_notes = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
NOTES = [n for n in _notes]


def log_x(coordinates, base=2):
    """
    Do the log_base 'base' of coordinates, where coordinates is the x, y pairs of coordinates.
        Only performs log on the y values. The x values remain the same
    :param coordinates: the values to take the log of
    :param base: the base of the logarithm to take
    :return: log base 'base' of the y values in coordinates, while leaving the x values the same
    """
    return np.array([coordinates[:, 0], np.log(coordinates[:, 1]) / np.log(base)]).T


def get_amp(path, num_samples, length, measurement, slicing=(0, 0)):
    """
    Returns a measurement of the amplitude. Different measurements are available:
        - 'avg': the average value
        - 'avg_abs': the average of the absolute values
        - 'max': the max value
        - 'max_abs': the max of the absolute values
    :param path: the path to the audio.wav file
    :param num_samples: the number of samples to use over the whole audio
    :param length: the length of the 3d print. Used to put the points at the right x value
    :param measurement: what measurement to do
    :param slicing: how to slice the audio file in seconds. IE: a slice of [1, 1] will start computing values at
        1 second into the recording, and stop 1 second before the end
    :return: a numpy array of x,y coordinate pairs
    """
    measurement = measurement.lower()

    fs, data = wavfile.read(path)
    data = data[fs * slicing[0]:len(data) - fs * slicing[1]]

    # Check if there are multiple input channels. If so, average them
    if len(data.shape) != 1:
        data = np.reshape(np.average(data, axis=1), [-1, ])

    # If our measurement requires doing the absolute value, do it
    if measurement in ['avg_abs', 'max_abs']:
        data = np.abs(data)
        measurement = measurement.replace("_abs", '')

    # Need to do this in order to make it work for some reason
    data = data.copy(order="C")

    ret = []

    # The size of each bucket based on num_samples, and how far along the x-axis to increment each measurement by
    bucket_size = len(data) / float(num_samples)
    length_inc = length / float(num_samples)

    for i in range(num_samples - 1):
        # The start and end of this bucket
        start, end = int(i * bucket_size), int((i + 1) * bucket_size)

        # Perform the necessary measurement on the data and add it to the return values
        if measurement == 'avg':
            v = np.average(data[start:end])
        elif measurement == 'max':
            v = np.max(data[start:end])
        else:
            raise ValueError("Unknown measurement: %s" % measurement)
        ret.append([i * length_inc, v])

    return np.array(ret)


def get_single_note(path, num_samples, length, note, num, fs=44100, overtones=False, undertones=False, start_freq=20.0,
                    max_freq=22050, slicing=(0, 0), tune=440.0):
    """
    Returns a measurement of how much a note is present compared to the rest of the audio in each chunk. Each 'note'
        is considered to be a range around the note, going from the logarithmic middle between this note and the
        previous, to the logarithmic middle between this note and the next.
        Notes are calculated based on the normal 12-tone scale.
        Over/Undertones can be taken into account with the overtones and undertones parameters.

    :param path: the path to the audio.wav file
    :param num_samples: the number of samples to use over the whole audio
    :param length: the length of the 3d print. Used to put the points at the right x value
    :param note: what note to measure
    :param num: which octave to find this note (eg: 4 would look in the 4th octave on a piano)
    :param fs: the sampling frequency to use for the fft
    :param overtones: if true, also takes into account all overtones of the given note
    :param undertones: if true, also takes into account all undertones of the given note
    :param start_freq: the starting frequency in Hz of the fourier data. Defaults to 20Hz since humans normally can't
        hear much lower
    :param max_freq: the highest frequency to use for overtones. Defaults to 22050 since humans can't hear higher
    :param slicing: how to slice the audio file in seconds. IE: a slice of [1, 1] will start computing values at
        1 second into the recording, and stop 1 second before the end
    :param tune: the frequency to tune the A above middle C (A4) to. Most common is A=440Hz
    :return: a numpy array of x,y coordinate pairs
    """
    # Fix some inputs
    note, num, tune, start_freq = _check_note(note, num, tune, start_freq=start_freq)

    # Read in the audio data
    fs, data = wavfile.read(path)
    data = data[fs * slicing[0]:len(data) - fs * slicing[1]]

    # Check there is enough data to do stuff
    if len(data) < num_samples * fs:
        raise ValueError("There is not enough data to do get_single_note. Record more data or lower num_samples")

    # Check if there are multiple input channels. If so, average them
    if len(data.shape) != 1:
        data = np.reshape(np.average(data, axis=1), [-1, ])

    # Need to do this in order to make it work for some reason
    data = data.copy(order="C")

    ret = []

    for i in range(num_samples):
        # The start and end of this bucket
        start, end = int(len(data) / float(num_samples) * i), int(len(data) / float(num_samples) * (i + 1))

        # Do the sliding window fft
        slice = np.abs(np.fft.rfft(data[start:start + fs]))
        count = 1
        start += fs
        while start + fs < end:
            slice += np.abs(np.fft.rfft(data[start:start + fs]))
            count += 1
            start += fs
        slice /= count
        #slice = np.abs(np.fft.rfft(data[start:end]))

        # Build all of the ranges of values
        ranges = [get_note_bounds(note, num, tune=tune)]

        # Build all of the ranges for the overtones, up until the lowest value goes over the length of data
        if overtones:
            t_note, t_num = note, num
            while True:
                t_note, t_num = shift_note(t_note, t_num, 12, tune=tune)
                new_b = get_note_bounds(t_note, t_num, tune=tune)
                if new_b[0] > len(slice) or new_b[0] > max_freq:
                    break
                ranges.append(new_b)

        # Build all of the ranges for the undertones, up until the highest value goes under the start_freq
        if undertones:
            t_note, t_num = note, num
            while True:
                t_note, t_num = shift_note(t_note, t_num, -12, tune=tune)
                new_b = get_note_bounds(t_note, t_num, tune=tune)
                if new_b[0] < start_freq:
                    break
                ranges.append(new_b)

        # Sum all of the values of data within the bounds
        total = 0
        for bounds in ranges:
            total += np.sum(slice[int(bounds[0]):int(bounds[1])])

        v = total / np.sum(slice)
        ret.append([i * length / float(num_samples), v])

    return np.array(ret)


def get_range(path, num_samples, length, ran, fs=44100, slicing=(0, 0), tune=440.0):
    """
    Returns a measurement of how much a frequency range is present compared to the rest of the audio in each chunk.
    :param path: the path to the audio.wav file
    :param num_samples: the number of samples to use over the whole audio
    :param length: the length of the 3d print. Used to put the points at the right x value
    :param fs: the sampling frequency to use for the fft
    :param slicing: how to slice the audio file in seconds. IE: a slice of [1, 1] will start computing values at
        1 second into the recording, and stop 1 second before the end
    :param tune: the frequency to tune the A above middle C (A4) to. Most common is A=440Hz
    :return: a numpy array of x,y coordinate pairs
    """
    # Fix some inputs
    if not isinstance(ran[0], (int, float, np.int_, np.float_)) \
            or not isinstance(ran[1], (int, float, np.int_, np.float_)):
        raise TypeError("ran must be a tuple with 2 frequency values")
    if ran[1] <= ran[0]:
        raise ValueError("ran[0] must be < ran[1], instead: %s" % (ran, ))

    # Read in the audio data
    fs, data = wavfile.read(path)
    data = data[fs * slicing[0]:len(data) - fs * slicing[1]]

    # Check there is enough data to do stuff
    if len(data) < num_samples * fs:
        raise ValueError("There is not enough data to do get_range. Record more data or lower num_samples")

    # Check if there are multiple input channels. If so, average them
    if len(data.shape) != 1:
        data = np.reshape(np.average(data, axis=1), [-1, ])

    # Need to do this in order to make it work for some reason
    data = data.copy(order="C")

    ret = []

    for i in range(num_samples):
        # The start and end of this bucket
        start, end = int(len(data) / float(num_samples) * i), int(len(data) / float(num_samples) * (i + 1))

        # Do the sliding window fft
        slice = np.abs(np.fft.rfft(data[start:start + fs]))
        count = 1
        start += fs
        while start + fs < end:
            slice += np.abs(np.fft.rfft(data[start:start + fs]))
            count += 1
            start += fs
        slice /= count
        #slice = np.abs(np.fft.rfft(data[start:end]))

        # Sum all of the values of data within the bounds
        total = np.sum(slice[int(ran[0]):int(ran[1])])

        v = total / np.sum(slice) / float(ran[1] - ran[0])
        ret.append([i * length / float(num_samples), v])

        """
        slice = slice[20:] / np.max(slice)
        plt.clf()
        plt.xscale('log')
        plt.plot(range(len(slice)), slice)
        plt.pause(0.01)
        """

    return np.array(ret)


def get_note_bounds(note, num, tune=440.0):
    """
    Returns the range in Hz corresponding to the logarithmic middle between this note and the previous, to the
        logarithmic middle between this note and the next.
        Notes are calculated based on the normal 12-tone scale.
    :param note: what note to get the range of
    :param num: which octave to find this note (eg: 4 would look in the 4th octave on a piano)
    :param tune: the frequency to tune the A above middle C (A4) to. Most common is A=440Hz
    :return: the bounds in Hz for the given note
    """
    # Fix some inputs
    note, num, tune = _check_note(note, num, tune)

    # Get the frequencies
    p_note, p_num = shift_note(note, num, half_steps=-1, tune=tune)
    n_note, n_num = shift_note(note, num, half_steps=1, tune=tune)
    prev_f = get_note_freq(p_note, p_num, tune=tune)
    this_f = get_note_freq(note, num, tune=tune)
    next_f = get_note_freq(n_note, n_num, tune=tune)

    return (prev_f * this_f) ** 0.5, (this_f * next_f) ** 0.5


def get_note_freq(note, num, tune=440.0):
    """
    Returns the frequency of the given note at the given octave range.
        Notes are calculated based on the normal 12-tone scale.
    :param note: what note to get frequency of
    :param num: which octave to find this note (eg: 4 would look in the 4th octave on a piano)
    :param tune: the frequency to tune the A above middle C (A4) to. Most common is A=440Hz
    :raises TypeError: if num is not an integer
    :raises ValueError: if tune <= 0, or if note is not a recognized note
    :return: the frequency of the given note
    """
    # Fix some inputs
    note, num, tune = _check_note(note, num, tune)

    # Find the frequency of the A in this octave
    freq = tune * (2 ** (num - 4)) if num != 4 else tune

    # Shift to the correct note (make sure to shift the index by -9 to move tuning to A4)
    return freq * 2 ** ((_notes.index(note) - 9) / float(len(_notes)))


def shift_note(note, num, half_steps, tune=440.0):
    """
    Returns the note shifted by the specified number of half_steps as a tuple (str, int) == (note, num).
        Notes are calculated based on the normal 12-tone scale.
    :param note: what note to shift
    :param num: which octave to find this note (eg: 4 would look in the 4th octave on a piano)
    :param half_steps: the number of half-steps to shift this note
    :param tune: the frequency to tune the A above middle C (A4) to. Most common is A=440Hz
    :raises TypeError: if num is not an integer
    :raises ValueError: if tune <= 0, or if note is not a recognized note
    :return: the frequency of the given note
    """
    # Fix some inputs
    note, num, tune = _check_note(note, num, tune)

    # Shift num to the correct octave
    shift = _notes.index(note) + half_steps
    num += int(shift / len(_notes)) if shift >= 0 else math.floor(shift / len(_notes))
    note = _notes[(_notes.index(note) + half_steps) % len(_notes)]

    return note, num


def _check_note(note, num, tune, start_freq=None):
    """
    Raises errors if note, num, or tune are bad values. Converts note, num, and tune into better values and returns
        them as a tuple.
    :param note: the note
    :param num: which octave to find this note (eg: 4 would look in the 4th octave on a piano)
    :param tune: the frequency to tune the A above middle C (A4) to. Most common is A=440Hz
    :param start_freq: the starting frequency in Hz of the fourier data. Defaults to 20Hz since humans normally can't
        hear much lower
    :return: (note.lower(), num, float(tune)) if values are able to be used
    """
    tune = float(tune)
    if not isinstance(num, (int, np.int_)):
        raise TypeError("num must be an integer, instead is: %s" % type(num))
    if tune <= 0:
        raise ValueError("tune must be > 0, instead is: %d" % tune)
    note = note.lower()
    if note not in _notes:
        raise ValueError("Unknown note: %s" % note)
    if start_freq is not None and start_freq <= 1:
        raise ValueError("start_freq must be > 1, instead is: %d" % start_freq)

    # Return these new values
    ret = [note, num, tune]
    if start_freq is not None:
        ret.append(start_freq)

    return ret


def audio_vis(fft=True, rate=44100, chunk_size=4096, start_cutoff=20):
    """
    Opens a window with a matplotlib plot of the current microphone audio data being recorded. This is used to check
        that the fft is working properly.
    :param fft: if True, a fast fourier transform is done on the incoming data
    :param rate: the frequency of data recording in Hz
    :param chunk_size: the number of samples to take every chunk
    :param start_cutoff: if doing fft, the starting frequency to print to the graph in Hz. Defaults to 20 because that
        is around the frequency humans can begin hearing sounds
    """

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True)

    max_height = 100

    while True:

        # Read in data from the audio stream
        data = stream.read(chunk_size)
        ints = np.array(
            [int.from_bytes(data[i * 2:(i + 1) * 2], byteorder='big', signed=True) for i in range(chunk_size)])

        # Clear the last plot
        plt.clf()

        # If we are doing fft on data, do it. Normalize to be between 0 and 1. Cutoff the values smaller
        #   than start_cutoff. Set the ylim to be [0, 1] and the xscale to be log
        if fft:
            ints = np.abs(np.fft.rfft(ints)[start_cutoff + 1:])
            ints /= max(ints)
            plt.ylim(0, 1)
            plt.xscale('log')

        # Otherwise set the ylim to be the max_height
        else:
            # Set the new max height if need be
            m = max(abs(ints))
            if m > max_height:
                max_height = m
            plt.ylim((-max_height, max_height))

        plt.plot(np.linspace(start_cutoff, len(ints), len(ints)), ints)
        plt.pause(0.0001)

        # Exit once window is closed
        if not plt.get_fignums():
            break
