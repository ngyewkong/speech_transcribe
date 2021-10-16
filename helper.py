# Helper Functions to record audio for use in streamlit 
# code base is from https://github.com/datarootsio/rootslab-streamlit-demo/blob/master/helper.py


import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write


def create_spectrogram(voice_sample):
    """
    Creates and saves a spectrogram plot for a sound sample.
    Parameters:
        voice_sample (str): path to sample of sound
    Return:
        fig
    """

    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    plt.subplot(211)
    plt.title(f"Spectrogram of file {voice_sample}")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(212)
    plt.specgram(original_wav, Fs=sampling_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=10, fs=16000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(myrecording, fs):
    write('audio.wav', fs, myrecording)
    return None