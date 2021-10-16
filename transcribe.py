# Import Required Libraries
import torch
import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load Pre-Trained Model 
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

file_name = './audio.wav'
data = wavfile.read(file_name)
framerate = data[0]
sound_data = data[1]
time = np.arange(0, len(sound_data))/framerate
print("Sampling Rate: ", framerate,"Hz")

# Resampling audio file to fit Facebook's Model Sampling Rate of 16000Hz
def resample(file_name):
    input_audio, _ = librosa.load(file_name, sr=16000)
    return input_audio, _

# Transcribe
def transcribe(input_audio, saved_model=model, saved_tokenizer=tokenizer):
    input_values = saved_tokenizer(input_audio, return_tensors="pt").input_values
    logits = saved_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = saved_tokenizer.batch_decode(predicted_ids)[0]
    return transcription
    