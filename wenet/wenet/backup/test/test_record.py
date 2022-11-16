import pyaudio
import wave
import os
from scipy.io import wavfile
import numpy as np

from tqdm import tqdm
import moviepy.editor as mp
import webrtcvad
import collections
import sys
import signal
import pyaudio

from array import array
import time


video_path = r"/media/sfy/File/我的视频/myhome.mkv"
def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    my_clip.audio.write_audiofile(f'{videos_file_path}.wav')
# extract_audio(video_path)



def record_audio(wave_out_path,record_second):
  CHUNK = 1024
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 16000
  p = pyaudio.PyAudio()
  stream = p.open(format=FORMAT,
          channels=CHANNELS,
          rate=RATE,
          input=True,
          frames_per_buffer=CHUNK)
  wf = wave.open(wave_out_path, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  print("* recording")
  for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
    data = stream.read(CHUNK)
    wf.writeframes(data)
  print("* done recording")
  stream.stop_stream()
  stream.close()
  p.terminate()
  wf.close()

record_audio("/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_20s.wav",record_second=20)

