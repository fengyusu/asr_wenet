
'''
客户端
'''


import sys
import json
import ast
import pyaudio
import wave
from tqdm import tqdm
import torch
import datetime
from datetime import timedelta as td
import time
print(torch.cuda.is_available())

import wenetruntime as wenet

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

wav_file = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output.wav"
wav_file_10s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_10s.wav"
wav_file_20s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_20s.wav"
wav_file_60s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_60s.wav"
wav_file_myhome = r"/media/sfy/File/我的视频/myhome.mkv.wav"
wav_file_60s_sep = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output/output_60s/vocals.wav"
wav_file_video_test = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/test_video.wav"

class asr_grpc_client:
    def __init__(self, hostname="127.0.0.1", port="10087", nbest=2, continuous_decoding=True):
        self.host = hostname
        self.port = port
        self.nbest = nbest
        self.continuous_decoding = continuous_decoding
        self.done = False

    def start_recognize(self):
        self.done = False
        self.asr_tast = threading.Thread(target=self.run)
        self.asr_tast.start()

    def end_recognize(self):
        self.done = True
        print("end1")
        self.asr_tast.join()
        print("end2")

    def bidirectional_streaming_method(self):

        def denoise(raw_data, sample_rate=16000):
            pad_list = [0, 1, -4, 10, -16, 15, 2, -58, 783, 1362, 1207, 1278, 1281, 1254, 1218, 1087, 987, 1000, 989,
                        999,
                        979, 960, 1136, 1187, 1057, 1060, 1135, 1258, 1430, 1425, 1195, 1008, 825, 696, 669, 581, 562,
                        668,
                        793, 923, 1055, 1024, 882, 762, 646, 524, 443, 378, 311, 341, 383, 392, 331, 192, 158, 284, 393,
                        337, 168, 129, 145, 87, 31, -162, -486, -668, -641, -556, -449, -402, -377, -257, -179, -268,
                        -370,
                        -499, -665, -742, -708, -571, -498, -553, -714, -992, -1163, -1098, -1045, -1017, -961, -853,
                        -730,
                        -651, -580, -584, -735, -1006, -1151, -1099, -1045, -1065, -1139, -1274, -1368, -1312, -1218,
                        -1164,
                        -1098, -1019, -956, -961, -949, -955, -993, -976, -928, -844, -745, -703, -676, -633, -658,
                        -761,
                        -823, -719, -597, -516, -390, -328, -388, -476, -559, -559, -407, -267, -281, -288, -267, -322,
                        -339, -297, -307, -238, -111, 4, 115, 218, 243, 190, 186, 230, 326, 418, 543, 581, 462, 351,
                        302,
                        237, 319, 599, 828, 981, 1003, 867, 721, 688, 584, 474, 586, 748, 812, 815, 869, 856, 887, 956,
                        905,
                        931, 1101, 1188, 1090, 1012, 1043, 1018, 1026, 1057, 994, 959, 934, 917, 838, 745, 720, 749,
                        882,
                        883, 792, 832, 936, 985, 1000, 1048, 1045, 1066, 1091, 1068, 1022, 861, 661, 554, 537, 511, 456,
                        430, 440, 534, 665, 640, 636, 710, 590, 311, 34, -165, -139, 127, 368, 455, 468, 375, 304, 385,
                        345,
                        165, 41, -80, -236, -327, -427, -547, -609, -650, -635, -530, -472, -486, -567, -646, -557,
                        -410,
                        -252, -195, -410, -615, -726, -833, -827, -681, -634, -737, -801, -851, -942, -926, -819, -707,
                        -633, -590, -603, -705, -885, -1069, -1155, -1159, -1155, -1215, -1312, -1291, -1189, -1058,
                        -975,
                        -964, -924, -910, -896, -909, -881, -805, -827, -837, -827, -850, -836, -766, -761, -840, -899,
                        -903, -819, -625, -574, -634, -663, -646, -470, -244, -79, -40, -148, -271, -380, -495, -455,
                        -275,
                        -219, -317, -332, -228, -61, 129, 149, 83, 68, 66, 101, 155, 252, 308, 245, 230, 294, 441, 533,
                        517,
                        547, 549, 601, 752, 843, 773, 667, 623, 530, 460, 445, 420, 461, 623, 746, 787, 841, 854, 819,
                        784,
                        762, 811, 930, 948, 919, 924, 932, 951, 966, 961, 942, 849, 769, 794, 817, 862, 871, 764, 675,
                        682,
                        709, 737, 849, 947, 1054, 1149, 1102, 905, 701, 503, 371, 350, 372, 364, 418, 531, 529, 518,
                        539,
                        568, 615, 654, 588, 384, 216, 181, 192, 242, 291, 171, 39, 6, 66, 141, 112, 102, 135, 89, -92,
                        -242,
                        -328, -430, -478, -511, -488, -403, -360, -411, -532, -618, -561, -421, -412, -463, -507, -555,
                        -575, -655, -738, -827, -889, -846, -770, -662, -592, -604, -652, -690, -620, -618, -763, -928,
                        -1045, -1078, -1101, -1185, -1261, -1239, -1251, -1101, -883, -830, -777, -754, -726, -650,
                        -631,
                        -696, -833, -846, -694, -588, -619, -796, -937, -920, -806, -621, -487, -491, -515, -432, -356,
                        -344, -368, -321, -175, -128, -121, -120, -172, -204, -138, -101, -79, -42, -72, -29, -27, -9,
                        87,
                        184, 273, 313, 332, 337, 345, 394, 458, 452, 447, 532, 611, 603, 449, 376, 596, 810, 951, 1067,
                        1065, 1008, 983, 932, 817, 791, 767, 662, 577, 610, 739, 918, 978, 807, 736, 819, 940, 1053,
                        1118,
                        1171, 1169, 1147, 1062, 1018, 1052, 1005, 924, 860, 807, 815, 880, 945, 875, 813, 939, 1022,
                        1025,
                        1009, 1004, 1100, 1207, 1197, 1044, 790, 537, 407, 523, 638, 637, 608, 555, 593, 751, 863, 760,
                        637,
                        562, 435, 341, 149, -20, -40, -51, 3, 105, 108, 102, 88, 91, 105, 148, 229, 164, 28, -179, -359,
                        -401, -429, -541, -728, -775, -773, -829, -817, -805, -821, -710, -595, -591, -597, -649, -836,
                        -976, -935, -809, -638, -572, -597, -730, -806, -718, -705, -809, -886, -886, -925, -998, -1117,
                        -1229, -1198, -1157, -1253, -1396, -1500, -1490, -1359, -1180, -1064, -919, -724, -699, -719,
                        -615,
                        -611, -770, -891, -927, -835, -717, -699, -799, -808, -737, -702, -650, -646, -626, -569, -551,
                        -564, -542, -452, -364, -260, -179, -171, -246, -306, -236, -229, -216, -204, -241, -105, 60,
                        154,
                        211, 284, 326, 220, 226, 353, 404, 402, 363, 274, 166, 253, 392, 351, 373, 586, 717, 839, 1083,
                        1111, 1058, 1083, 971, 905, 997, 993, 975, 1054, 1054, 1007, 1054, 1065, 1016, 968, 969, 972,
                        984,
                        1040, 1161, 1268, 1306, 1298, 1175, 991, 902, 821, 774, 960, 1177, 1220, 1163, 1047, 1035, 1146,
                        1195, 1100, 1003, 1066, 1109, 1174, 1163, 1012, 818, 639, 540, 507, 554, 579, 593, 651, 672,
                        715,
                        787, 876, 889, 746, 657, 581, 449, 294, 115, 70, 115, 223, 286, 214, 97, 26, -45, -93, -52, -42,
                        -140, -195, -191, -258, -357, -419, -486, -564, -592, -565, -488, -422, -406, -358, -316, -435,
                        -622, -656, -640, -661, -707, -837, -893, -776, -688, -738, -817, -788, -728, -746, -801, -865,
                        -952, -966, -969, -1066, -1087, -1083]
            padding_data = np.array(pad_list, dtype=np.int16)
            # padding_num = len(padding_data)

            raw_padding_data = np.concatenate([padding_data, raw_data])
            processed_padding_data = logmmse.logmmse(data=raw_padding_data, sampling_rate=sample_rate)
            return processed_padding_data[-len(raw_data):]

        # create a generator
        def request_messages(wav_file=wav_file_60s):

            data, sr = sf.read(wav_file, dtype='int16')
            # data = logmmse.logmmse(data=data, sampling_rate=sr)


            sample_rate = 16000
            interval = 0.5
            sample_interval = int(sample_rate * interval)

            for i in range(-sample_interval, len(data), sample_interval):
                if i == -sample_interval:
                    dec_conf = pb2.Request.DecodeConfig(nbest_config=self.nbest,
                                                        continuous_decoding_config=self.continuous_decoding)
                    request = pb2.Request(decode_config=dec_conf)
                    yield request
                    time.sleep(0.1)
                else:
                    # chunk_data = denoise(data[i: min(i + sample_interval, len(data))]).tobytes()
                    chunk_data = data[i: min(i + sample_interval, len(data))].tobytes()
                    request = pb2.Request(audio_data=chunk_data)
                    yield request
                    time.sleep(0.1)

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

            total_chunk = int(sample_rate / chunk_size * record_second)
            for i in range(0, total_chunk + 1):
                chunk_data = stream.read(chunk_size)
                request = pb2.Request(audio_data=chunk_data)
                yield request
                time.sleep(0.01)

        all_segment_text = []
        all_segment_wordieces = []
        self.asr_result = ""
        self.asr_seg_result = []
        last_tamp = 0

        response_iterator = self.stub.Recognize(request_messages())
        for response in response_iterator:
            if self.done:
                break
            if response.nbest and response.status == 0:
                cur_segment_text = response.nbest[0].sentence.replace('</context>', '').replace('<context>', '')
                if all_segment_text and all_segment_text[-1] == cur_segment_text:
                    continue


                is_seg_final = response.type == pb2.Response.Type.final_result
                if is_seg_final and cur_segment_text:
                    # print('\n final_result  '+cur_segment_text+'\n')
                    all_segment_text.append(cur_segment_text)
                    cur_segment_text = ""
                    print(all_segment_text, len(all_segment_text[-1]))

                    cur_segment_wordpieces = []
                    for wp in response.nbest[0].wordpieces:
                        cur_segment_wordpieces.append((wp.word, wp.start, wp.end, wp.start-last_tamp))
                        last_tamp = wp.end
                    all_segment_wordieces.append(cur_segment_wordpieces)
                    print(all_segment_wordieces, len(all_segment_wordieces[-1]))

                    assert len(all_segment_text[-1]) == len(all_segment_wordieces[-1])

                total_text = "".join(all_segment_text) + "" + cur_segment_text
                # if total_text:
                #     sys.stdout.write('\r' + total_text)
                #     sys.stdout.flush()

                # print("cur_segment_text ", cur_segment_text)
                # print("wordpiece ", response.nbest[0].wordpieces)

                self.asr_result = total_text
                self.asr_seg_result = all_segment_text + ([] if cur_segment_text == "" else [cur_segment_text])


        edit_dis = minDistance(self.asr_result, label)
        print("\nedit_dis ", edit_dis)
        self.asr_result = ""

    def run(self):
        print("\n--------------start recorgnize---------------")
        with grpc.insecure_channel("{}:{}".format(self.host, self.port)) as channel:
            self.stub = pb2_grpc.ASRStub(channel)
            self.bidirectional_streaming_method()
        print("\n--------------end recorgnize---------------")


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



if __name__ == '__main__':
    client = asr_grpc_client()
    client.start_recognize()
    exit()


