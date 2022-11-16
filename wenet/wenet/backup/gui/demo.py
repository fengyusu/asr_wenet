# here put the import lib
import sys
import os
import time
from datetime import datetime, date, timedelta

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.Qt import *
from PyQt5 import QtMultimedia
from PyQt5.QtCore import QUrl


from ui import Ui_MainWindow


import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from multiprocessing import Process, Queue
import threading
import grpc
import soundfile as sf
import pyaudio
import wave
import moviepy.editor as mp

import wenet_pb2 as pb2
import wenet_pb2_grpc as pb2_grpc

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告
import logging
import os.path
import sys
import multiprocessing
import gensim
import jieba
import torch
from torch.utils.data import *
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd



class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
        self.is_voice = None

def audio_generator(file_or_realtime=None,
                    frame_duration_ms=500, sample_rate=16000, channels=1,
                    record_second=20):

    frame_duration_s = frame_duration_ms / 1000.0
    frame_byte_size = int(sample_rate * frame_duration_s * 2)

    def yield_frame_data(audio):
        offset = 0
        timestamp = 0.0
        while offset + frame_byte_size < len(audio):
            yield Frame(audio[offset : offset + frame_byte_size], 0.0, frame_duration_s)
            timestamp += frame_duration_s
            offset += frame_byte_size


    if file_or_realtime is None:

        frame_duration_s = frame_duration_ms / 1000.0
        chunk_size = int(sample_rate * frame_duration_s)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)
        chunk_size = int(sample_rate * frame_duration_ms / 1000.)
        total_chunk = int(sample_rate / chunk_size * record_second)
        timestamp = datetime.now()

        for i in range(0, total_chunk + 1):
            chunk_data = stream.read(chunk_size)
            yield Frame(chunk_data, 0.0, frame_duration_s)

        stream.stop_stream()
        stream.close()
        p.terminate()

    else:
        raw_data, sr = sf.read(file_or_realtime, dtype=np.int16)
        assert sr == sample_rate
        audio = raw_data.tobytes()

        for frame in yield_frame_data(audio):
            yield frame
            time.sleep(frame_duration_s)


class asr_grpc_client:
    def __init__(self, hostname="127.0.0.1", port="10087", nbest=1, continuous_decoding=True):
        self.host = hostname
        self.port = port
        self.nbest = nbest
        self.continuous_decoding = continuous_decoding
        self.asr_task = None
        self.done = False
        self.init_asr_result()

    def init_asr_result(self):
        self.asr_rt_result = ""
        self.asr_seg_result = []
        self.asr_seg_wordieces = []
        self.final_seg_count = 0

    """ 
    recgnize_type  1: recognize from wav_file
                   2: recognize from pyaudio
    """
    def start_recognize(self, wav_file=None, recognize_type=2):
        self.done = False
        self.wav_file = wav_file
        self.recognize_type = recognize_type
        if not wav_file:
            self.recognize_type = 2

        self.init_asr_result()

        self.asr_task = threading.Thread(target=self.run)
        self.asr_task.start()

    def end_recognize(self):
        self.done = True
        self.wav_file = None

        # self.init_asr_result()
        # if self.asr_task is not None:
        #     self.asr_task.join()
        #     self.asr_task = None

    def bidirectional_streaming_method(self):

        # create a generator
        def request_messages(wav_file = self.wav_file):

            data, sr = sf.read(wav_file, dtype='int16')

            sample_rate = 16000
            interval = 0.5
            sample_interval = int(sample_rate * interval)

            for i in range(-sample_interval, len(data), sample_interval):
                if i == -sample_interval:
                    dec_conf = pb2.Request.DecodeConfig(nbest_config=self.nbest,
                                                        continuous_decoding_config=self.continuous_decoding)
                    request = pb2.Request(decode_config=dec_conf)
                    yield request
                    time.sleep(interval)
                else:
                    chunk_data = data[i: min(i + sample_interval, len(data))].tobytes()
                    request = pb2.Request(audio_data=chunk_data)
                    yield request
                    time.sleep(interval)

        def request_messages_realtime(record_second=60):

            dec_conf = pb2.Request.DecodeConfig(nbest_config=self.nbest,
                                                continuous_decoding_config=self.continuous_decoding)
            request = pb2.Request(decode_config=dec_conf)
            yield request

            sample_rate = 16000
            interval = 0.5
            chunk_size = int(sample_rate * interval)

            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1,
                            rate=sample_rate, input=True,
                            frames_per_buffer=chunk_size)

            if self.wav_file is not None:
                wf = wave.open(self.wav_file, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)

            total_chunk = int(sample_rate / chunk_size * record_second)
            for i in range(0, total_chunk + 1):
                chunk_data = stream.read(chunk_size)
                request = pb2.Request(audio_data=chunk_data)
                yield request

                if self.wav_file is not None:
                    wf.writeframes(chunk_data)
                time.sleep(0.01)

            stream.stop_stream()
            stream.close()
            p.terminate()
            if self.wav_file is not None:
                wf.close()

        def request_messages_generator(record_second=60):

            sample_rate = 16000
            channels = 1
            frame_duration_ms = 500

            dec_conf = pb2.Request.DecodeConfig(nbest_config=self.nbest,
                                                continuous_decoding_config=self.continuous_decoding)
            request = pb2.Request(decode_config=dec_conf)
            yield request

            generator = audio_generator(file_or_realtime=None if self.recognize_type==2 else self.wav_file,
                                        frame_duration_ms=frame_duration_ms, sample_rate=sample_rate, channels=channels,
                                        record_second=record_second)

            is_record = self.wav_file is not None and self.recognize_type==2
            if is_record:
                wf = wave.open(self.wav_file, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)

            for frame in generator:
                frame_byte_date = frame.bytes
                request = pb2.Request(audio_data=frame_byte_date)
                yield request
                if is_record:
                    wf.writeframes(frame_byte_date)
                # time.sleep(0.005)

            if is_record:
                wf.close()


        all_segment_text = []
        all_segment_wordpieces = []

        if self.recognize_type in [1,2]:
            response_iterator = self.stub.Recognize(request_messages_generator())
        else:
            return

        for response in response_iterator:
            if self.done:
                break
            if response.nbest and response.status == 0:
                cur_segment_text = response.nbest[0].sentence.replace('</context>', '').replace('<context>', '')
                if (all_segment_text and all_segment_text[-1] == cur_segment_text) or not cur_segment_text:
                    continue

                cur_segment_wordpieces = []
                if response.nbest[0].wordpieces:
                    for wp in response.nbest[0].wordpieces:
                        if self.asr_seg_wordieces or cur_segment_wordpieces:
                            prev_wordpiece = self.asr_seg_wordieces[-1][-1] if not cur_segment_wordpieces else \
                                cur_segment_wordpieces[-1]
                            prev_wordpiece[4] = wp.start - prev_wordpiece[2]
                            cur_segment_wordpieces.append([wp.word, wp.start, wp.end, prev_wordpiece[4], 0])
                        else:
                            cur_segment_wordpieces.append([wp.word, wp.start, wp.end, 3600000, 0])
                    assert len(cur_segment_text) == len(cur_segment_wordpieces)

                is_seg_final = response.type == pb2.Response.Type.final_result
                if is_seg_final:
                    self.final_seg_count += 1
                    all_segment_text.append(cur_segment_text)
                    cur_segment_text = ""

                    all_segment_wordpieces.append(cur_segment_wordpieces)
                    cur_segment_wordpieces = []

                    # cur_wordpiece = [(wp.word, wp.start, wp.end) for wp in response.nbest[0].wordpieces]
                    # self.asr_seg_timestamp.append(cur_wordpiece)

                total_text = "".join(all_segment_text) + "" + cur_segment_text
                # if total_text:
                #     sys.stdout.write('\r' + total_text)
                #     sys.stdout.flush()

                self.asr_rt_result = total_text
                self.asr_seg_result = (all_segment_text + [cur_segment_text]) if cur_segment_text != "" else all_segment_text
                self.asr_seg_wordieces = (all_segment_wordpieces + [cur_segment_wordpieces]) if cur_segment_wordpieces else all_segment_wordpieces


    def run(self):
        print("\r--------------start recorgnize---------------")
        with grpc.insecure_channel("{}:{}".format(self.host, self.port)) as channel:
            self.stub = pb2_grpc.ASRStub(channel)
            self.bidirectional_streaming_method()
        print("\r--------------end recorgnize---------------")


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers, batch_size, device='cpu',
                 vocab_size=2):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = device

        self.word_embeds = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

        self.to(self.device)

    def init_embedding(self, embedding):
        self.word_embeds.weight = nn.Parameter(embedding)
        self.word_embeds.weight.requires_grad = False

    def init_hidden(self):
        return (torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2).to(self.device),
                torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim // 2).to(self.device))

    def forward(self, inputs):
        embeddings = self.word_embeds(inputs)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        tag_scores = self.fc(lstm_out)
#         tag_scores = F.softmax(tag_space, dim=-1)
        return tag_scores

tag_punctuation = ['X', '，', '。', '！', '？', '、', '：']
unknown_word = '<UNK>'
padding_word = '<PAD>'

init_w2v_path = '/media/sfy/Study/graduation/PostProcess/model/sgns.wiki.word.bz2'
vocab_file = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/vocab.txt"
emb_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/w2v_embedding.npy"
lstm_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/wiki_200w_1/bilstm_20_32.final.pt"
lstm_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/wiki_158w_infrequent/bilstm_2_32.final.pt"

class PunctuationAppender:
    def __init__(self):
        self.init_vocab()
        self.init_model()

    def init_vocab(self):
        self.vocab = []
        with open(vocab_file) as f:
            for word in f.readlines():
                word = word.strip()
                if word:
                    self.vocab.append(word)
        print("vocab size ", len(self.vocab))

        self.word_to_idx = {v: i for i, v in enumerate(self.vocab)}
        self.tag_to_idx = {v: i for i, v in enumerate(tag_punctuation)}

    def init_model(self):
        batch_size = 1
        device = 'cpu'
        self.lstm = BiLSTM(input_dim=300, hidden_dim=160, output_dim=7, num_layers=3,
                      batch_size=1, device=device, vocab_size=len(self.vocab))

        self.lstm.load_state_dict(torch.load(lstm_model_path))
        self.lstm = self.lstm.to(device)
        print("device ", device)

        # w2v_model_path = '/media/sfy/Study/graduation/PostProcess/model/sgns.wiki.word.bz2'
        # self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, encoding="utf-8")

    def punc_predict(self, sentence):
        senseg = list(jieba.cut(sentence, cut_all=False))
        if not senseg:
            return ""

        xpara = [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx[unknown_word] for w in senseg]
        x = torch.LongTensor(xpara)
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            tag_scores = self.lstm(x)
            a, tag_index = torch.max(torch.squeeze(tag_scores, 0), dim=1)

        punc_res = ''
        for i in range(len(tag_index)):
            cur_punc = tag_punctuation[int(tag_index[i])]
            none_punc = tag_punctuation[0]
            if len(senseg[i]) == 1:
                punc_res += cur_punc
            else:
                punc_res += none_punc*(len(senseg[i])-1) + cur_punc

        assert len(punc_res) == len(sentence)
        return punc_res

    def append_punc_raw(self, sentence):
        if not sentence:
            return sentence
        punc_predict_res = self.punc_predict(sentence)
        res = ""
        assert len(sentence) == len(punc_predict_res)
        for i in range(len(sentence)):
            res += sentence[i]
            res += punc_predict_res[i] if punc_predict_res[i] != 'X' else ""
        return res

    def append_punc(self, asr_seg_result, asr_seg_wordieces, final_seg_count):
        total_sentence = "".join(asr_seg_result)
        total_len = len(total_sentence)
        if not total_sentence:
            return ""

        asr_all_wordieces = sum(asr_seg_wordieces, [])

        start_seg_idx = len(self.cache_res)
        end_seg_idx = len(asr_seg_result)
        cur_sentence = "".join(asr_seg_result[start_seg_idx:])
        cur_punc_predict_res = self.punc_predict(cur_sentence)
        cur_len = len(cur_sentence)

        text_res = ""
        total_idx = 0
        for seg_idx in range(start_seg_idx):
            for i in range(len(self.cache_res[seg_idx])):
                text_res += total_sentence[total_idx]
                text_res += self.cache_res[seg_idx][i] if self.cache_res[seg_idx][i] != tag_punctuation[0] else ""
                total_idx += 1

        cached_total_len = total_idx
        for seg_idx in range(start_seg_idx, end_seg_idx):
            cur_seg_len = len(asr_seg_result[seg_idx])
            cur_seg_cahce = []
            for i in range(cur_seg_len):
                text_res += total_sentence[total_idx]
                if cur_punc_predict_res[total_idx-cached_total_len] != tag_punctuation[0]:
                    if total_idx < len(asr_all_wordieces) and asr_all_wordieces[total_idx][4] < 100:
                        cur_seg_cahce.append(tag_punctuation[0])
                    else:
                        text_res += cur_punc_predict_res[total_idx-cached_total_len]
                        cur_seg_cahce.append(text_res[-1])
                elif total_idx < len(asr_all_wordieces) and asr_all_wordieces[total_idx][4] > 500:
                    text_res += "，"
                    cur_seg_cahce.append(text_res[-1])
                else:
                    cur_seg_cahce.append(tag_punctuation[0])
                total_idx += 1
            if seg_idx < final_seg_count:
                self.cache_res.append(cur_seg_cahce)

        # text_res += asr_seg_result[-1] if asr_seg_result and final_seg_count != len(asr_seg_result) else ""

        return text_res

    def append_punc_v1(self, asr_seg_result, asr_seg_wordieces, final_seg_count):
        total_sentence = "".join(asr_seg_result)
        total_len = len(total_sentence)
        if not total_sentence:
            return ""


        cache_seg_num = len(self.cache_res)
        if 0 < cache_seg_num < len(asr_seg_wordieces) and asr_seg_wordieces[cache_seg_num]:
            if self.cache_res[-1] and self.cache_res[-1][-1] not in tag_punctuation[1:]:
                if asr_seg_wordieces[cache_seg_num][0][3] > 800:
                    self.cache_res[-1] += '。'
                elif asr_seg_wordieces[cache_seg_num][0][3] > 300:
                    self.cache_res[-1] += '，'
        cache_res = "".join(self.cache_res)

        start_seg_idx = cache_seg_num
        end_seg_idx = len(asr_seg_result)

        cur_segs_len = list(map(len, asr_seg_result[start_seg_idx:end_seg_idx]))
        cur_sentence = "".join(asr_seg_result[start_seg_idx:])
        cur_punc_predict_res = self.punc_predict(cur_sentence)
        cur_punc_predict_seg_res = []
        cur_seg_start_idx = 0
        for l in cur_segs_len:
            cur_punc_predict_seg_res.append(cur_punc_predict_res[cur_seg_start_idx:cur_seg_start_idx+l])
            cur_seg_start_idx += l

        res_text = cache_res

        for seg_idx in range(start_seg_idx, end_seg_idx):
            cur_seg_punc_predict = cur_punc_predict_seg_res[seg_idx-cache_seg_num]

            if seg_idx < len(asr_seg_wordieces):
                cur_seg_wordpieces = asr_seg_wordieces[seg_idx]
            else:
                cur_seg_wordpieces = []

            cur_seg_raw_text = asr_seg_result[seg_idx]
            cur_seg_res_text = ""

            assert len(cur_seg_raw_text) == len(cur_seg_punc_predict)
            assert len(cur_seg_raw_text) == len(cur_seg_wordpieces)

            for i,c in enumerate(cur_seg_raw_text):
                cur_seg_res_text += c
                if (cur_seg_punc_predict[i] in ['，', '、', '：'] and (cur_seg_wordpieces and cur_seg_wordpieces[i][4] > 300)) or \
                    (cur_seg_punc_predict[i] in ['。', '！', '？'] and (cur_seg_wordpieces and cur_seg_wordpieces[i][4] > 600)):
                    cur_seg_res_text += cur_seg_punc_predict[i]
                elif (cur_seg_wordpieces and cur_seg_wordpieces[i][4] > 400):
                    cur_seg_res_text += "，"

            res_text += cur_seg_res_text
            if cache_seg_num <= seg_idx < end_seg_idx-1 or final_seg_count==len(asr_seg_result):
                self.cache_res.append(cur_seg_res_text)

        return res_text

    def clear(self):
        self.cache_res = []



class Asr_Demo(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Asr_Demo, self).__init__()
        self.setupUi(self)

        self.init()
        self.photo_flag = 0
        self.start_time = 0
        self.cur_state = 0

        self.asr = asr_grpc_client()
        self.punc_pre = PunctuationAppender()

    def init(self):

        self.player = QMediaPlayer()

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.show_camera_image)

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.show_video)

        self.audio_timer = QTimer()
        self.audio_timer.timeout.connect(self.show_audio)

        self.asr_result_timer = QTimer()
        self.asr_result_timer.timeout.connect(self.show_asr_result)

        self.openCameraButton.clicked.connect(self.open_camera)

        self.openAudioButton.clicked.connect(self.open_audio)

        self.openVideoButton.clicked.connect(self.open_video)

        self.recordButton.clicked.connect(self.recored_save)

        self.closeButton.clicked.connect(self.close)
        self.exitButton.clicked.connect(self.exit)

        '''
        天依蓝 #66ccff
        初音绿 #66ffcc
        言和绿 #99ffff
        阿绫红 #ee0000
        双子黄 #ffff00
        '''
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint )
        self.setWindowOpacity(0.98)
        self.bgColor = QColor(0, 50, 50, 80)
        self.openCameraButton.setStyleSheet('''QWidget{border-radius:7px;background-color:#66ffcc;}''')
        self.openAudioButton.setStyleSheet('''QWidget{border-radius:7px;background-color:#66ffcc;}''')
        self.openVideoButton.setStyleSheet('''QWidget{border-radius:7px;background-color:#66ffcc;}''')
        self.recordButton.setStyleSheet('''QWidget{border-radius:7px;background-color:#66ffcc;}''')
        self.closeButton.setStyleSheet('''QWidget{border-radius:7px;background-color:#66ffcc;}''')
        self.exitButton.setStyleSheet('''QWidget{border-radius:7px;background-color:#66ffcc;}''')

        self.label.setStyleSheet('''QWidget{border-radius:7px;background-color:#66ccff;}''')
        self.textBrowser.setStyleSheet('''QWidget{border-radius:7px;background-color:#99ffff;}''')

        self.label.setScaledContents(True)  # 图片自适应

    def open_camera(self, is_record=False):
        self.close()
        self.cur_state = 1
        self.cap = cv2.VideoCapture(0)

        self.open_audio(audio_src=2, from_file=None, is_record=is_record)

        self.camera_timer.start(40)  # 每40毫秒读取一次，即刷新率为25帧
        self.show_camera_image()

    def show_camera_image(self):
        def paint_chinese_opencv( img, text, left, top, textColor=(0, 255, 0), textSize=20):
            if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 创建一个可以在给定图像上绘图的对象
            draw = ImageDraw.Draw(img)
            # 字体的格式
            font = "AaMingYueJiuLinTian.ttf"
            fontStyle = ImageFont.truetype(font, textSize, encoding="utf-8")
            # 绘制文本
            draw.text((left, top), text, textColor, font=fontStyle)
            # 转换回OpenCV格式
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        flag, self.image = self.cap.read()  # 从视频流中读取图片
        image_raw = cv2.resize(self.image, (640, 480))
        image_raw = cv2.flip(image_raw, 1)

        h, w  = image_raw.shape[:2]

        # display_words_size = 10
        # subtitle = self.asr.asr_result
        # cur_start_idx = -len(subtitle)%display_words_size
        # sub_pos_left = int(0.05 * w)
        # sub_pos_top = int(0.85 * h)
        # sub_size = int(0.07 * h)
        # image_show = self.paint_chinese_opencv(image_show, subtitle[cur_start_idx:], left=sub_pos_left, top=sub_pos_top,
        #                                        textColor=(200, 20, 20), textSize=sub_size)

        image_show = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        self.showImage = QtGui.QImage(image_show.data, w, h, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(self.showImage))
        # self.label.setScaledContents(True) #图片自适应

        if self.cur_state == 4:
            print("recored ")
            self.lz.write(image_raw)


    def open_video(self):
        self.close()
        self.cur_state = 2
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '../',)
        if not fname:
            self.cur_state = 0
            return
        self.cap_video = cv2.VideoCapture(fname)

        audio_file = os.path.splitext(fname)[0] + '.wav'
        if not os.path.exists(audio_file):
            print(audio_file)
            my_clip = mp.VideoFileClip(fname)
            # my_clip.audio.write_audiofile(audio_file)
            my_clip.audio.write_audiofile(audio_file, fps=16000, ffmpeg_params=["-ac","1"])

        # self.v_wf = wave.open(audio_file, 'rb')
        # self.v_p = pyaudio.PyAudio()
        # self.v_stream = self.v_p.open(format=self.v_p.get_format_from_width(self.v_wf.getsampwidth()),
        #                           channels=self.v_wf.getnchannels(),
        #                           rate=self.v_wf.getframerate(),
        #                           output=True)
        # print(self.v_wf.getnchannels(), self.v_wf.getframerate())
        #
        # self.asr.start_recognize(wav_file=audio_file, recognize_type=1)
        self.open_audio(audio_src=1, from_file=audio_file)

        self.interval = 50
        self.video_timer.start(self.interval)
        self.show_video()

    def show_video(self):
        flag, self.image = self.cap_video.read()
        image_show = cv2.resize(self.image, (1280, 720))
        width, height = image_show.shape[:2]
        image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
        # image_show = cv2.flip(image_show, 1)
        self.showImage = QtGui.QImage(image_show.data, height, width, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(self.showImage))

        # text = self.asr.asr_result
        # self.textBrowser.setPlainText(text)
        #
        # CHUNK = int(self.v_wf.getframerate() * self.interval / 1000.)
        # data = self.v_wf.readframes(CHUNK)
        # if data != '':
        #     self.v_stream.write(data)

    # 0:choose audio file    1:use user define audio file    2:from sound
    def open_audio(self, audio_src=0, from_file=None, is_record=False):
        if audio_src == 2:
            if is_record:
                self.asr.start_recognize(wav_file=self.audio_file, recognize_type=2)
            else:
                self.asr.start_recognize(wav_file=None, recognize_type=2)
        else:

            if audio_src == 1:
                fname = from_file
            else:
                self.close()
                self.cur_state = 3
                fname, _ = QFileDialog.getOpenFileName(self, '选择音频文件', '../', )

            if not fname:
                self.cur_state = 0
                return

            self.wf = wave.open(fname, 'rb')
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                                channels=self.wf.getnchannels(),
                                rate=self.wf.getframerate(),
                                output=True)

            self.asr.start_recognize(wav_file=fname, recognize_type=1)

            # self.audio_interval = 50
            # self.audio_timer.start(self.audio_interval)
            # self.show_audio()

            file = QUrl.fromLocalFile(fname)
            content = QtMultimedia.QMediaContent(file)
            self.player.setMedia(content)
            self.player.setVolume(50.0)
            self.player.play()


        self.asr_result_interval = 50
        self.asr_result_timer.start(self.asr_result_interval)
        self.show_asr_result()


    def show_audio(self):

        CHUNK = int(self.wf.getframerate() * self.audio_interval*1. / 1000.)
        data = self.wf.readframes(CHUNK)
        if data != '':
            self.stream.write(data)


    def show_asr_result(self):
        text = self.asr.asr_rt_result
        # print("asr timer time ", time.time() - self.start_time)
        # self.start_time = time.time()

        asr_seg_result = self.asr.asr_seg_result
        asr_seg_wordieces = self.asr.asr_seg_wordieces
        final_seg_count = self.asr.final_seg_count
        # print(self.asr.asr_seg_result)
        # print(self.asr.asr_seg_wordieces)
        # print(self.asr.final_seg_count, len(self.asr.asr_seg_result), len(self.asr.asr_seg_wordieces))
        text = self.punc_pre.append_punc_v1(asr_seg_result, asr_seg_wordieces, final_seg_count)

        # if text:
        #     sys.stdout.write('\r' + text)
        #     sys.stdout.flush()
        self.textBrowser.setPlainText(text)
        # print("punc time ", time.time() - self.start_time)



    def recored_save(self):
        self.close()

        self.save_path = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/record_video/"
        self.video_file = os.path.join(self.save_path, "tmp_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.lz = cv2.VideoWriter(self.video_file, fourcc, 20, (640, 480))

        self.audio_file = os.path.join(self.save_path, "tmp_audio.wav")
        self.video_audio_file = os.path.join(self.save_path, "record_0.mp4")

        self.open_camera(is_record=True)
        self.cur_state = 4


    def close(self):
        if self.cur_state == 1:
            self.camera_timer.stop()  # 停止读取
            self.cap.release()  # 释放摄像头
            print("close 1")

        elif self.cur_state == 2:
            self.video_timer.stop()
            self.cap_video.release()
            self.audio_timer.stop()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            print("close 2")

        elif self.cur_state == 3:
            self.audio_timer.stop()
            self.asr.end_recognize()
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            print("close 3")

        elif self.cur_state == 4:
            self.camera_timer.stop()
            self.cap.release()
            self.lz.release()
            ad = mp.AudioFileClip(self.audio_file)
            vd = mp.VideoFileClip(self.video_file)
            vd2 = vd.set_audio(ad)
            vd2.write_videofile(self.video_audio_file)

            print("close 4")

        self.label.clear()
        self.textBrowser.clear()

        self.player.stop()
        self.asr_result_timer.stop()
        self.asr.end_recognize()
        self.punc_pre.clear()

        self.start_time = 0
        self.cur_state = 0

    def closeEvent(self, event):
        self.close()
        print("Force close!")

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = QPoint(e.x(), e.y())
            self._tracking = True

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None


    def exit(self):
        self.close()
        app = QApplication.instance()
        app.quit()
        print("Force exit!")


if __name__ == '__main__':

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    demo = Asr_Demo()
    demo.show()
    sys.exit(app.exec_())

# s =  "我以与父亲不相见已二年余了我最不能忘记的是他的背影那年冬天祖母死了父亲的差使也交谢了正是祸不单行的日子我从北京到徐州打算跟着父亲奔丧回家到徐州见着父亲看见满院狼藉的东西又想起祖母不禁簌簌地流下眼泪父亲说事已如此不易难过好在天无绝人之路回家变卖点质父亲还了亏空又借钱办了丧事这些日子家中光景很是惨淡一半为了丧事一半为了父亲赋闲丧事完毕父亲要到南京谋事我也要回北京念书我们便同行到南京时有朋友约去逛街勾留了一日第二日上午"
# pa = PunctuationAppender()
# res = pa.append_punc_raw(s)
# print(res)
