import torch
import random
from glob import glob
from omegaconf import OmegaConf
from src.silero.utils import (init_jit_model, 
                       split_into_batches,
                       read_audio,
                       read_batch,
                       prepare_model_input)
from colab_utils import (record_audio,
                         audio_bytes_to_np,
                         upload_audio)

# imports for uploading/recording
import numpy as np
import ipywidgets as widgets
from scipy.io import wavfile
from IPython.display import Audio, display, clear_output
from torchaudio.functional import vad

class SpeechtoText():
    def __init__(self, record_seconds = 4):
        self.device = torch.device('cpu')   # you can use any pytorch device
        self.models = OmegaConf.load('models.yml')
        self.language = "English" 
        self.model, self.decoder = init_jit_model(self.models.stt_models.en.latest.jit, device=self.device)
        self.record_seconds = record_seconds
        self.sample_rate = 16000

    # wav to text method
    def wav_to_text(self, f='test.wav'):
        batch = read_batch([f])
        input = prepare_model_input(batch, device=self.device)
        output = self.model(input)
        return self.decoder(output[0].cpu())

    def _recognize(self, audio):
        display(Audio(audio, rate=self.sample_rate, autoplay=True))
        wavfile.write('test.wav', self.sample_rate, (32767*audio).numpy().astype(np.int16))
        transcription = self.wav_to_text()
        print('\n\nTRANSCRIPTION:\n')
        print(transcription)
        return(transcription)

    def _record_audio(self, b):
        clear_output()
        audio = record_audio(self.record_seconds)
        wavfile.write('recorded.wav', self.sample_rate, (32767*audio).numpy().astype(np.int16))
        return self._recognize(audio)

    def _upload_audio(self, b):
        clear_output()
        audio = upload_audio()
        self._recognize(audio)
        return audio

    # button = widgets.Button(description="Record Speech")
    # button.on_click(_record_audio)
    # display(button)
    def run(self):       
        text = self._record_audio(1)
        print("the text is: ",text)
        print(type(text))
        return text

    # 'text' is result of speech-to-text 