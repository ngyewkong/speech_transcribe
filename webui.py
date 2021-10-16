# Import Required Libraries

from librosa.core import audio
import streamlit as st
import numpy as np
from pathlib import Path
import soundfile as sf
import os
import glob
from helper import create_spectrogram, read_audio, record, save_record
import transcribe

st.title("Speech-to-Text Transcribing Tool")
st.sidebar.title("About")
st.sidebar.info("This web application is a Speech-to-Text Transcribing Tool. It uses Facebook's pre-trained model to do Natural Language Processing to transcribe audio. Do take note that only English is currently supported.")
st.write("Record your speech in English")

filename = st.text_input("Choose a filename: ")
option = st.radio(
    "Upload or Record an audio wav file", ('Upload', 'Record') 
)

if option == 'Upload':
    uploaded_file = st.file_uploader("Choose a wav file", type='wav')
    if uploaded_file is not None:
        st.audio(read_audio(uploaded_file))
        input_audio, _ = transcribe.resample(uploaded_file)
            
        fig = create_spectrogram(uploaded_file)
        st.pyplot(fig, clear_figure=True)
        transcribed_text = transcribe.transcribe(input_audio)
        st.write("Transcribed Text: ",transcribed_text)

else:
    if st.button("Click to Record"):
        duration = 10 
        fs = 22000
        with st.spinner('Recording in Progress. Please wait.'):
            myrecording = record(duration, fs)
            record_state = st.info("Saving sample")
            path_myrecording = "./audio.wav"
                
            save_record(myrecording, fs)
            record_state.success("Done! Saved sample")
        st.audio(read_audio(path_myrecording))
        input_audio, _ = transcribe.resample(path_myrecording)
            
        fig = create_spectrogram(path_myrecording)
        st.pyplot(fig, clear_figure=True)
        transcribed_text = transcribe.transcribe(input_audio)
        st.write("Transcribed Text: ",transcribed_text)

        
