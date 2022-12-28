
'''
客户端
'''


import sys
import json
import ast
import string
import pyaudio
import wave
from tqdm import tqdm
import torch
import datetime
from datetime import timedelta as td
import time
print(torch.cuda.is_available())

import websockets
import websocket

import threading
import soundfile as sf
import logmmse
import webrtcvad
import asyncio
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import optuna

import grpc
import wenet_pb2 as pb2
import wenet_pb2_grpc as pb2_grpc

import time
from concurrent import futures

import logging
import os.path
import sys
import operator
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

from pycorrector.macbert.macbert_corrector import MacBertCorrector
from pycorrector.ernie_csc.ernie_csc_corrector import ErnieCSCCorrector

wav_file = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output.wav"
wav_file_10s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_10s.wav"
wav_file_20s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_20s.wav"
wav_file_60s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_60s.wav"
wav_file_myhome = r"/media/sfy/File/我的视频/myhome.mkv.wav"
wav_file_60s_sep = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output/output_60s/vocals.wav"
wav_file_video_test = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/test_video/test_video.wav"

use_wav_file = wav_file_60s


def minDistance(word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)
        if (m == 0 and n == 0) or word1 == word2:
            return 0
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            dp[i][0] = i
        for j in range(1, n+1):
            dp[0][j] = j

        for i in range(1, m+1):
            for j in range(1, n+1):
                c1 = word1[i-1]
                c2 = word2[j-1]
                if c1 == c2:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

        return dp[m][n]
label = "我以与父亲不相见已二年余了我最不能忘记的是他的背影那年冬天祖母死了父亲的差使也交谢了正是祸不单行的日子我从北京到徐州打算跟着父亲奔丧回家到徐州见着父亲看见满院狼藉的东西又想起祖母不禁簌簌地流下眼泪父亲说事已如此不易难过好在天无绝人之路回家变卖点质父亲还了亏空又借钱办了丧事这些日子家中光景很是惨淡一半为了丧事一半为了父亲赋闲丧事完毕父亲要到南京谋事我也要回北京念书我们便同行到南京时有朋友约去逛街勾留了一日第二日上午"
def displayWaveform(data, sr=16000): # 显示语音时域波形

    plt.subplot(2,1,1)
    time = np.arange(0, len(data[0])) * (1.0 / sr)
    plt.plot(time, data[0])

    plt.subplot(2, 1, 2)
    time = np.arange(0, len(data[1])) * (1.0 / sr)
    plt.plot(time, data[1])

    plt.title("wave")
    plt.xlabel("time")
    plt.ylabel("h")

    plt.show()


class MacBertCorrector(object):
    def __init__(self, model_dir, prun_sim_dict, tag_punctuation):
        self.name = 'macbert_corrector'
        t1 = time.time()
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_dir = "shibing624/macbert4csc-base-chinese"
            logger.warning(f'local model {bin_path} not exists, use default HF model {model_dir}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def pinyin_sim_filter(self, target, candidates, context, target_idx=0):
        res_sim = []
        for c in candidates.split():
            if c not in self.word_to_idx or target not in self.word_to_idx or c == target:
                continue

            cur_cos_sim = self.cos_sim[self.word_to_idx[c]][self.word_to_idx[target]]

            cur_cand_pinyin = lazy_pinyin(c, style=Style.TONE3)[0]
            cur_target_pinyin = lazy_pinyin(target, style=Style.TONE3)[0]
            # print(cur_cand_pinyin, cur_target_pinyin)
            cur_pinyin_dis = self.pinyin_edit_distance(cur_cand_pinyin, cur_target_pinyin)
            cur_pinyin_sim = 10.0 if cur_pinyin_dis < 1e-3 else 1. / cur_pinyin_dis

            cur_ngram_sim = 0
            for k in range(2, 5):
                cur_word = c + context[target_idx + 1:target_idx + k]
                if cur_word in self.wc_dict:
                    cur_sim = np.log10(self.wc_dict[cur_word])
                    cur_ngram_sim = max(cur_ngram_sim, cur_sim)
                if target_idx - k >= 0:
                    cur_word = context[target_idx - k:target_idx] + c
                    if cur_word in self.wc_dict:
                        cur_sim = np.log10(self.wc_dict[cur_word])
                        cur_ngram_sim = max(cur_ngram_sim, cur_sim)
            if cur_pinyin_sim > 0.6 and cur_ngram_sim > 1e-5:
                res_sim.append([c, cur_cos_sim, cur_pinyin_sim, cur_ngram_sim])

        res_sim.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
        # print(res_sim)

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
                                                     context=context, target_idx=i)
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
    def __init__(self, lstm_model_path, vocab_file, tag_punctuation, w2v_model_path=None):
        self.vocab_file = vocab_file
        self.tag_punctuation = tag_punctuation
        self.lstm_model_path = lstm_model_path
        self.w2v_model_path = w2v_model_path
        self.unknown_word = '<UNK>'
        self.padding_word = '<PAD>'



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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.use_punc_predictor = False
        self.use_lm_corrector = False

        self.tag_punctuation = ['X', '，', '。', '！', '？', '、', '：']
        # self.init_w2v_path = '/media/sfy/Study/graduation/PostProcess/model/sgns.wiki.word.bz2'
        # self.emb_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/w2v_embedding.npy"
        self.vocab_file = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/vocab.txt"
        self.lstm_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/wiki_200w_1/bilstm_20_32.final.pt"
        self.lstm_model_path = "/media/sfy/Study/graduation/PostProcess/ChinesePunctuationPredictor/model/wiki_158w_infrequent/bilstm_2_32.final.pt"

        self.corrector_model_path = "/media/sfy/Study/graduation/PostProcess/model/macbert4csc-base-chinese"
        self.cossim_w2i_wc_file = "/media/sfy/Study/graduation/PostProcess/pycorrector/dict/cossim_w2i_wc.pkl"

        self.asr_client = AsrGrpcClient()
        if self.use_punc_predictor:
            self.punc_predictor = PunctuationAppender(lstm_model_path=self.lstm_model_path,
                                                      vocab_file=self.vocab_file, tag_punctuation=self.tag_punctuation)
        if self.use_lm_corrector:
            self.lm_corrector = MacBertCorrector(model_dir=self.corrector_model_path,
                                                 prun_sim_dict=self.cossim_w2i_wc_file, tag_punctuation=self.tag_punctuation)

        self.reset_subtitle_param()



    def reset_subtitle_param(self, max_subtitle_len = 23, asr_timer_interval = 500):
        self.subtitle = ""
        self.subtitle_cache = []
        self.last_pos = 0
        self.remain_times = 0
        self.max_subtitle_len = max_subtitle_len
        self.asr_timer_interval = asr_timer_interval

    def start_generate(self, recognize_type=1, wav_file=use_wav_file, use_inner_timer = False):

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
        text = self.asr_client.asr_rt_result
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
        print("text ", text)


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


if __name__ == '__main__':

    asr_subtitle_generator = AsrSubtitleGenerator()

    asr_subtitle_generator.start_generate(use_inner_timer=True)
    # time.sleep(10)
    # asr_subtitle_generator.end_generate()

    test_text = "我以与父亲不相见一二年余了我最不能忘记的是他的背影那年冬天祖母死了父亲的差事也交卸了正是祸不单行的日子我从北京到徐州打算跟着父亲奔丧回家到徐州按照父亲看见满月狼藉的东西又想起祖母不禁簌簌地流下眼泪父亲说事已如此不会难过好在天无绝人之路回家变卖田志父亲怀了亏空又借钱办了丧事这些日子加工光景很是惨败一半为了丧事一半为了父亲付钱丧事完毕父亲要到南京谋事我也要回北京念书我们便同行到南京时有朋友约去逛街勾留了一日第二日"
    print(minDistance(test_text, label))




