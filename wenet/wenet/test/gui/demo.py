# here put the import lib
import sys
import os
import time
import string
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
import operator
import sys



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

from pycorrector import config
# from pycorrector.utils.tokenizer import split_text_by_maxlen
from pypinyin import *
import pickle
from loguru import logger
from transformers import BertTokenizerFast, BertForMaskedLM



wav_file = "/test_data/output.wav"
wav_file_10s = "/test_data/output_10s.wav"
wav_file_20s = "/test_data/output_20s.wav"
wav_file_60s = "/test_data/output_60s.wav"
wav_file_myhome = r"/media/sfy/File/我的视频/myhome.mkv.wav"
wav_file_60s_sep = "/test_data/output/output_60s/vocals.wav"
wav_file_video_test = "/test_data/test_video/test_video.wav"

use_wav_file = wav_file_60s



class MacBertCorrector(object):
    def __init__(self, model_dir, prun_sim_dict, tag_punctuation, device):
        self.name = 'macbert_corrector'
        t1 = time.time()
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_dir = "shibing624/macbert4csc-base-chinese"
            logger.warning(f'local model {bin_path} not exists, use default HF model {model_dir}')

        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model = BertForMaskedLM.from_pretrained(model_dir)
        self.model.to(self.device)
        # logger.debug("Use device: {}".format(self.device))
        # logger.debug('Loaded macbert4csc model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

        self.unk_tokens = [' ', '“', '”', '‘', '’', '\n', '…', '—', '擤', '\t', '֍', '玕', '']
        self.tag_punctuation = tag_punctuation
        self.total_punctuation = list(string.punctuation) + self.tag_punctuation

        self.cossim_w2i_wc_file = prun_sim_dict
        self.init_prun_sim_filter()


    def get_errors(self, corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if i >= len(corrected_text):
                break
            if ori_char in self.unk_tokens:
                # deal with unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def init_prun_sim_filter(self):
        def save_pickle(file_name, data):
            f = open(file_name, "wb")
            pickle.dump(data, f)
            f.close()

        def load_pickle(file_name):
            f = open(file_name, "rb+")
            data = pickle.load(f)
            f.close()
            return data

        self.cos_sim, self.word_to_idx, self.wc_dict = load_pickle(self.cossim_w2i_wc_file)
        self.word_to_idx['[MASK]'] = len(self.word_to_idx)

    def pinyin_edit_distance(self, word1: str, word2: str):
        m = len(word1)
        n = len(word2)
        if (m == 0 and n == 0) or word1 == word2:
            return 0
        if m == 0 or n == 0:
            return max(m, n)

        tone1 = tone2 = 1
        if word1[-1].isdigit():
            m -= 1
            tone1 = int(word1[-1])
        if word2[-1].isdigit():
            n -= 1
            tone2 = int(word2[-1])
        tone_dis = abs(tone1 - tone2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c1 = word1[i - 1]
                c2 = word2[j - 1]
                if c1 == c2:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        prun_dis = dp[m][n]
        return prun_dis + tone_dis * 0.3

    def pinyin_sim_filter(self, target, candidates, context, target_idx=0, use_pinyin_sim=True):
        res_sim = []
        for i, c in enumerate(candidates.split(), start=1):
            if c not in self.word_to_idx or target not in self.word_to_idx or c == target:
                continue

            if use_pinyin_sim:
                cur_cos_sim = self.cos_sim[self.word_to_idx[c]][self.word_to_idx[target]]

                cur_cand_pinyin = lazy_pinyin(c, style=Style.TONE3)[0]
                cur_target_pinyin = lazy_pinyin(target, style=Style.TONE3)[0]
                # print(cur_cand_pinyin, cur_target_pinyin)
                cur_pinyin_dis = self.pinyin_edit_distance(cur_cand_pinyin, cur_target_pinyin)
                cur_pinyin_sim = 10.0 if cur_pinyin_dis < 1e-3 else 1. / cur_pinyin_dis
                order_score = 1.0
            else:
                cur_cos_sim = 1.0
                cur_pinyin_sim = 10.0
                order_score = np.log10(1.0+1.0/i)

            cur_ngram_sim = 0
            for k in range(2, 5):
                cur_word = c + context[target_idx + 1:target_idx + k]
                if cur_word in self.wc_dict:
                    cur_sim = np.log10(self.wc_dict[cur_word])
                    cur_ngram_sim = max(cur_ngram_sim, cur_sim)
                if target_idx - k >= 0:
                    cur_word = context[target_idx-k+1:target_idx] + c
                    if cur_word in self.wc_dict:
                        cur_sim = np.log10(self.wc_dict[cur_word])
                        cur_ngram_sim = max(cur_ngram_sim, cur_sim)
            if cur_pinyin_sim > 0.6 or not use_pinyin_sim:
                res_sim.append([c, cur_cos_sim, cur_pinyin_sim, cur_ngram_sim, order_score])

        res_sim.sort(key=lambda x: x[1] * x[2] * x[3] * x[4], reverse=True)
        print(res_sim)

        return res_sim[0][0] if res_sim and res_sim[0][3] > 1e-5 else target

    def split_text_by_maxlen(self, text, maxlen=512):
        result = []
        for i in range(0, len(text), maxlen):
            result.append((text[i:i + maxlen], i))
        return result

    def macbert_correct_pinyin(self, text, threshold=0.9, verbose=True):
        """
        句子纠错
        :param text: 句子文本
        :param threshold: 阈值
        :param verbose: 是否打印详细信息
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        if not text:
            return text, None

        text_new = ''
        details = []
        # 长句切分为短句
        blocks = self.split_text_by_maxlen(text, maxlen=128)
        # print("block ", blocks)
        block_texts = [block[0] for block in blocks]
        inputs = self.tokenizer(block_texts, padding=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        for ids, (text, idx) in zip(outputs.logits, blocks):
            decode_tokens_new = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).split(' ')
            decode_tokens_old = self.tokenizer.decode(inputs['input_ids'][idx%128], skip_special_tokens=True).split(' ')
            if len(decode_tokens_new) != len(decode_tokens_old):
                continue
            probs = torch.max(torch.softmax(ids, dim=-1), dim=-1)[0].cpu().numpy()
            decode_tokens = ''
            for i in range(len(decode_tokens_old)):
                if probs[i + 1] >= threshold:
                    decode_tokens += decode_tokens_new[i]

                else:

                    if decode_tokens_old[i] not in self.word_to_idx or decode_tokens_old[i] in self.tag_punctuation:
                        decode_tokens += decode_tokens_old[i]
                    else:
                        values, indices = ids[i+1].topk(20, dim=-1, largest=True, sorted=True)
                        tokens = self.tokenizer.decode(indices, skip_special_tokens=True)

                        context = "".join(decode_tokens_new)
                        res = self.pinyin_sim_filter(target=decode_tokens_old[i], candidates=tokens,
                                                     context=context, target_idx=i,
                                                     use_pinyin_sim=decode_tokens_old[i] != self.tokenizer.mask_token)
                        # print("correct res ", decode_tokens_old[i], res, )

                        decode_tokens += res

            corrected_text = decode_tokens[:len(text)]
            corrected_text, sub_details = self.get_errors(corrected_text, text)
            text_new += corrected_text
            sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
            details.extend(sub_details)


        return text_new, details

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

class PunctuationAppender:
    def __init__(self, lstm_model_path, vocab_file, tag_punctuation, w2v_model_path=None, device='cpu'):
        self.vocab_file = vocab_file
        self.tag_punctuation = tag_punctuation
        self.lstm_model_path = lstm_model_path
        self.w2v_model_path = w2v_model_path
        self.unknown_word = '<UNK>'
        self.padding_word = '<PAD>'

        self.device = device

        self.init_vocab()
        self.init_model()
        self.cache_res = []


    def init_vocab(self):
        self.vocab = []
        with open(self.vocab_file) as f:
            for word in f.readlines():
                word = word.strip()
                if word:
                    self.vocab.append(word)
        # print("vocab size ", len(self.vocab))

        self.word_to_idx = {v: i for i, v in enumerate(self.vocab)}
        self.tag_to_idx = {v: i for i, v in enumerate(self.tag_punctuation)}

    def init_model(self):
        batch_size = 1

        self.lstm = BiLSTM(input_dim=300, hidden_dim=160, output_dim=7, num_layers=3,
                      batch_size=batch_size, device=self.device, vocab_size=len(self.vocab))

        self.lstm.load_state_dict(torch.load(self.lstm_model_path))
        self.lstm = self.lstm.to(self.device)

        # self.w2v_model_path = '/media/sfy/Study/graduation/PostProcess/model/sgns.wiki.word.bz2'
        # self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, encoding="utf-8")

    def punc_predict(self, sentence):
        senseg = list(jieba.cut(sentence, cut_all=False))
        if not senseg:
            return ""

        xpara = [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx[self.unknown_word] for w in senseg]
        x = torch.LongTensor(xpara)
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            x = x.to(self.device)
            tag_scores = self.lstm(x)
            _, tag_index = torch.max(torch.squeeze(tag_scores, 0), dim=1)

        punc_res = ''
        for i in range(len(tag_index)):
            cur_punc = self.tag_punctuation[int(tag_index[i])]
            none_punc = self.tag_punctuation[0]
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

    def append_punc_v1(self, asr_seg_result, asr_seg_wordieces, final_seg_count):
        total_sentence = "".join(asr_seg_result)
        total_len = len(total_sentence)
        if not total_sentence:
            return ""


        cache_seg_num = len(self.cache_res)
        if 0 < cache_seg_num < len(asr_seg_wordieces) and asr_seg_wordieces[cache_seg_num]:
            if self.cache_res[-1] and self.cache_res[-1][-1] not in self.tag_punctuation[1:]:
                if asr_seg_wordieces[cache_seg_num][0][3] > 800:
                    self.cache_res[-1] += '。'
                elif asr_seg_wordieces[cache_seg_num][0][3] > 400:
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

class AsrGrpcClient:
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

    class Frame(object):
        def __init__(self, bytes, timestamp, duration):
            self.bytes = bytes
            self.timestamp = timestamp
            self.duration = duration
            self.is_voice = None

    def audio_generator(self, file_or_realtime=None,
                        frame_duration_ms=500, sample_rate=16000, channels=1,
                        record_second=20):

        frame_duration_s = frame_duration_ms / 1000.0
        frame_byte_size = int(sample_rate * frame_duration_s * 2)

        def yield_frame_data(audio):
            offset = 0
            timestamp = 0.0
            while offset + frame_byte_size < len(audio):
                yield self.Frame(audio[offset: offset + frame_byte_size], 0.0, frame_duration_s)
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
                yield self.Frame(chunk_data, 0.0, frame_duration_s)

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

    """ 
    recgnize_type  1: recognize from wav_file
                   2: recognize from pyaudio
    """
    def start_recognize(self, recognize_type=2, wav_file=None, ):
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

            generator = self.audio_generator(file_or_realtime=None if self.recognize_type==2 else self.wav_file,
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


class AsrSubtitleGenerator:
    def __init__(self):
        self.timer_is_running = False
        self.use_punc_predictor = True
        self.use_lm_corrector = False

        self.tag_punctuation = ['X', '，', '。', '！', '？', '、', '：']
        # self.init_w2v_path = '/media/sfy/Study/graduation/PostProcess/model/sgns.wiki.word.bz2'
        # self.emb_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/w2v_embedding.npy"
        self.vocab_file = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/vocab.txt"
        self.lstm_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/wiki_200w_1/bilstm_20_32.final.pt"
        self.lstm_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/wiki_158w_infrequent/bilstm_2_32.final.pt"

        self.corrector_model_path = "/media/sfy/Study/graduation/PostProcess/model/macbert4csc-base-chinese"
        # self.corrector_model_path = "/media/sfy/Study/graduation/PostProcess/pycorrector/pycorrector/macbert/output/model/wiki/pytorch_export"
        self.cossim_w2i_wc_file = "/media/sfy/Study/graduation/PostProcess/pycorrector/dict/cossim_w2i_wc.pkl"


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.asr_client = AsrGrpcClient()
        if self.use_punc_predictor:
            self.punc_predictor = PunctuationAppender(lstm_model_path=self.lstm_model_path,
                                                  vocab_file=self.vocab_file, tag_punctuation=self.tag_punctuation, device = self.device)
        if self.use_lm_corrector:
            self.lm_corrector = MacBertCorrector(model_dir=self.corrector_model_path,
                                             prun_sim_dict=self.cossim_w2i_wc_file, tag_punctuation=self.tag_punctuation, device = self.device)

        self.reset_subtitle_param()


    def reset_subtitle_param(self, max_subtitle_len = 23, asr_timer_interval = 500):
        self.subtitle = ""
        self.subtitle_cache = []
        self.last_pos = 0
        self.remain_times = 0
        self.max_subtitle_len = max_subtitle_len
        self.asr_timer_interval = asr_timer_interval

    def start_generate(self, recognize_type=1, wav_file=use_wav_file,
                       use_inner_timer = False):

        if use_inner_timer:
            self.timer = threading.Timer(self.asr_timer_interval/1000., self.asr_timer,)
            self.timer.start()
            self.timer_is_running = True

        self.asr_client.start_recognize(recognize_type=recognize_type, wav_file=wav_file, )

    def end_generate(self):

        if self.timer_is_running:
            self.timer_is_running = False
            self.timer.join()

        self.asr_client.end_recognize()
        self.punc_predictor.clear()
        self.reset_subtitle_param()

    def asr_timer(self):
        if self.timer_is_running:
            self.generate_subtitle()
            self.timer = threading.Timer(self.asr_timer_interval/1000., self.asr_timer,)
            self.timer.start()

    def generate_subtitle(self):
        # text = self.asr_client.asr_rt_result
        # print("asr timer time ", time.time() - self.start_time)
        # self.start_time = time.time()

        asr_seg_result = self.asr_client.asr_seg_result
        asr_seg_wordieces = self.asr_client.asr_seg_wordieces
        final_seg_count = self.asr_client.final_seg_count
        # print("asr_seg_result ", asr_seg_result)
        # print(self.asr.asr_seg_result)
        # print(self.asr.asr_seg_wordieces)
        # print(self.asr.final_seg_count, len(self.asr.asr_seg_result), len(self.asr.asr_seg_wordieces))
        start_time = time.time()
        if self.use_punc_predictor:
            text = self.punc_predictor.append_punc_v1(asr_seg_result, asr_seg_wordieces, final_seg_count)
        punc_predictor_time = time.time() - start_time

        if self.use_lm_corrector:
            correct_raw_text = text[self.last_pos:]
            corrected_text,_ = self.lm_corrector.macbert_correct_pinyin(correct_raw_text)
            text = text[:self.last_pos] + corrected_text
        corrector_time = time.time() - start_time - punc_predictor_time
        # print("text ", text)


        cur_subtitle = text[self.last_pos:]
        while len(cur_subtitle) > self.max_subtitle_len or self.remain_times*self.asr_timer_interval>1400:
            pre_len = len(self.subtitle)
            # cur_len = min(len(cur_subtitle), self.max_subtitle_len)
            self.subtitle_cache.append(self.subtitle)
            self.last_pos += pre_len
            cur_subtitle = text[self.last_pos:]
            self.remain_times = 0

        # remove head puncturation
        if self.last_pos < len(text) and text[self.last_pos] in self.tag_punctuation:
            if self.subtitle_cache:
                self.subtitle_cache[-1] += text[self.last_pos]
            self.last_pos += 1
            cur_subtitle = cur_subtitle[1:]

        self.remain_times = self.remain_times+1 if self.subtitle == cur_subtitle else 0
        self.subtitle = cur_subtitle
        print(punc_predictor_time, corrector_time, self.subtitle)

        self.display_text = "\n".join(self.subtitle_cache[-25:] + [cur_subtitle])
        # self.textBrowser.setPlainText(display_text)

        return self.subtitle, self.display_text


class Asr_Demo(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Asr_Demo, self).__init__()
        self.setupUi(self)

        self.init_ui()
        self.photo_flag = 0
        self.start_time = 0
        self.cur_state = 0

        self.subtitle = ""
        self.subtitle_color = (255, 20, 20)

        self.asr = AsrSubtitleGenerator()


    def init_ui(self):

        self.player = QMediaPlayer()

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.show_camera_image)

        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.show_video)

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

    def paint_subtitle(self, img, text, textColor=(0, 255, 0)):
        h, w = img.shape[:2]
        sub_pos_left = int(0.05 * w)
        sub_pos_top = int(0.85 * h)
        sub_size = int(0.07 * h)

        # total_punc = list(string.punctuation) + [' ', '，', '。', '！', '？', '、', '：']
        # display_text = text[1:] if text and text[0] in total_punc else text
        display_text = text

        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        font = "AaMingYueJiuLinTian.ttf"
        font = "HanYiYanKaiW-2.ttf"
        fontStyle = ImageFont.truetype(font, sub_size, encoding="utf-8")
        draw.text((sub_pos_left, sub_pos_top), display_text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def open_camera(self, is_record=False):
        self.close()
        self.cur_state = 1
        self.cap = cv2.VideoCapture(0)

        self.open_audio(audio_src=2, from_file=None, is_record=is_record)

        self.camera_timer.start(40)  # 每40毫秒读取一次，即刷新率为25帧
        self.show_camera_image()

    def show_camera_image(self):

        flag, self.image = self.cap.read()  # 从视频流中读取图片
        image_raw = cv2.resize(self.image, (640, 480))
        image_raw = cv2.flip(image_raw, 1)
        h, w  = image_raw.shape[:2]
        image_show = self.paint_subtitle(image_raw, self.subtitle, textColor=self.subtitle_color)
        image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
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

        self.open_audio(audio_src=1, from_file=audio_file)

        self.interval = 50
        self.video_timer.start(self.interval)
        self.show_video()

    def show_video(self):
        flag, self.image = self.cap_video.read()
        image_show = cv2.resize(self.image, (1280, 720))
        h, w = image_show.shape[:2]
        # image_show = cv2.flip(image_show, 1)
        image_show = self.paint_subtitle(image_show, self.subtitle, textColor=self.subtitle_color)
        image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
        self.showImage = QtGui.QImage(image_show.data, w, h, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(self.showImage))

    # 0:choose audio file    1:use userdefine audio file    2:from sound
    def open_audio(self, audio_src=0, from_file=None, is_record=False):
        if audio_src == 2:
            self.asr.start_generate(recognize_type=2, wav_file=self.audio_file if is_record else None )
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

            self.asr.start_generate(recognize_type=1, wav_file=fname, )

            file = QUrl.fromLocalFile(fname)
            content = QtMultimedia.QMediaContent(file)
            self.player.setMedia(content)
            self.player.setVolume(50.0)
            self.player.play()


        self.asr_result_interval = 150
        self.asr.asr_timer_interval = self.asr_result_interval
        self.asr_result_timer.start(self.asr_result_interval)
        self.show_asr_result()


    def show_asr_result(self):
        self.subtitle, display_text = self.asr.generate_subtitle()

        self.textBrowser.setPlainText(display_text)
        # print("punc time ", time.time() - self.start_time)
        # if display_text:
        #     sys.stdout.write('\r' + display_text)
        #     sys.stdout.flush()



    def recored_save(self):
        self.close()

        self.save_path = "/test_data/record_video/"
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
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            print("close 2")

        elif self.cur_state == 3:
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
        self.asr.end_generate()
        self.subtitle = ""

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

# s =  "我以与父亲不相见已二年余了我最不能忘记的是他的背影"
# pa = PunctuationAppender()
# res = pa.append_punc_raw(s)
# print(res)
