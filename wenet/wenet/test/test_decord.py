
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
import asyncio
import librosa
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import logmmse
import optuna


wav_file = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output.wav"
wav_file_10s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_10s.wav"
wav_file_20s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_20s.wav"
wav_file_60s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_60s.wav"
wav_file_myhome = r"/media/sfy/File/我的视频/myhome.mkv.wav"
wav_file_60s_sep = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output/output_60s/vocals.wav"


# plt.figure(dpi=600) # 将显示的所有图分辨率调高
# plt.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus']=False # 显示符号


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


def nonstream_decord(input_file):
    decoder = wenet.Decoder(lang='chs', nbest=5)
    ans = decoder.decode_wav(input_file)
    print(ans)
nonstream_decord(wav_file_60s)

def stream_decord(input_file):

    with wave.open(input_file, 'rb') as fin:
        print(fin.getnchannels() )
        # assert fin.getnchannels() == 1
        wav = fin.readframes(fin.getnframes())

    decoder = wenet.Decoder(lang='chs', enable_timestamp=False,)

    # We suppose the wav is 16k, 16bits, and decode every 0.5 seconds
    interval = int(0.5 * 16000) * 2

    print("wav", len(wav), type(wav))
    # start_time = datetime.datetime.now()
    for i in range(0, len(wav), interval):

        last = False if i + interval < len(wav) else True
        chunk_wav = wav[i: min(i + interval, len(wav))]
        ans = decoder.decode(chunk_wav, last)

        if ans:
            ans_dict = ast.literal_eval(ans.replace("\n", ""))
            if ans_dict["type"] == "partial_result":
                # print(ans_dict["nbest"][0]["sentence"], end='',flush=True)
                # time.sleep(0.1)
                sys.stdout.write('\r'+ans_dict["nbest"][0]["sentence"])
                sys.stdout.flush()
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()
                print(ans_dict)

def realtime_decord(record_second=30):
    CHUNK = 8000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    decoder = wenet.Decoder(lang='chs', enable_timestamp=False, )


    print("* recording")
    total_chunk = int(RATE / CHUNK * record_second)
    for i in range(0, total_chunk+1):
        data = stream.read(CHUNK)
        # print("data ", len(data), type(data))

        last = False if i != total_chunk else True
        ans = decoder.decode(data, last)
        if ans:
            ans_dict = ast.literal_eval(ans.replace("\n", ""))
            if ans_dict["type"] == "partial_result":
                print(ans_dict["nbest"])
            else:
                print(ans_dict)


    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()



def test_websocket():
    WS_START = json.dumps({
        'signal': 'start',
        'nbest': 1,
        'continuous_decoding': True,
    })
    WS_END = json.dumps({
        'signal': 'end'
    })
    ws_uri = "ws://127.0.0.1:10086"

    def on_message(ws, message):
        pass
        # print("on_message", message)

    def on_data(ws, data, data_type, continue_):
        print("on_data", data, data_type, continue_)

    def on_cont_message(ws, data, continue_):
        print("on_cont_message", data)

    def on_error(ws, error):
        print(error)


    def on_close(ws, close_status_code, close_msg):
        print("### closed ###")


    def on_open(ws):

        def run(*args):
            while 1:

                print("send msg", type(data))

                ws.send(data.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                time.sleep(10)
                ws.send(WS_END, opcode=websocket.ABNF.OPCODE_TEXT)
                time.sleep(3)


            ws.close()


            print("thread terminating...")



        sample_rate = 16000
        data, sr = sf.read(wav_file, dtype='int16')
        ws.send(WS_START, opcode=websocket.ABNF.OPCODE_TEXT)
        time.sleep(1)
        a = threading.Thread(target=run, args=data)
        a.start()
        a.join()





    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(ws_uri,
                                on_message = on_message,
                                on_data= on_data,
                                on_cont_message=on_cont_message,
                                on_error = on_error,
                                on_close = on_close,
                                )

    ws.on_open = on_open

    ws.run_forever()


# test_websocket()

def request_server_decord(input_file=wav_file_60s_sep, ):

    WS_START = json.dumps({
        'signal': 'start',
        'nbest': 1,
        'continuous_decoding': True,
    })
    WS_END = json.dumps({
        'signal': 'end'
    })
    ws_uri = "ws://127.0.0.1:10086"

    async def ws_rec(data, ws_uri):
        begin = time.time()
        conn = await websockets.connect(ws_uri, ping_timeout=200)
        # print("connected ")
        # step 1: send start
        await conn.send(WS_START)
        ret = await conn.recv()
        # print(ret)
        # step 2: send audio data
        await conn.send(data)
        # step 3: send end
        await conn.send(WS_END)
        # step 4: receive result
        texts = []
        while 1:
            ret = await conn.recv()
            ret = json.loads(ret)
            if ret['type'] == 'final_result':
                nbest = json.loads(ret['nbest'])
                text = nbest[0]['sentence']
                texts.append(text)
            elif ret['type'] == 'speech_end':
                break
        # step 5: close
        try:
            await conn.close()
        except Exception as e:
            # this except has no effect, just log as debug
            # it seems the server does not send close info, maybe
            # print(e)
            pass
        time_cost = time.time() - begin
        return {
            'text': ''.join(texts),
            'time': time_cost,
        }


    sample_rate = 16000
    data, sr = sf.read(input_file, dtype='int16')
    # signal, sr = librosa.load(input_file, sr=None)
    # l = len(signal)
    # data = librosa.resample(signal[l//150*24:l//150*26], sr, sample_rate)
    print(sr)
    assert sr == sample_rate
    data_len = len(data)
    duration = data_len / sample_rate
    print(duration, type(data))

    processed_data = []
    interval = int(sample_rate * 0.3)


    pad_list = [0,1,-4,10,-16,15,2,-58,783,1362,1207,1278,1281,1254,1218,1087,987,1000,989,999,979,960,1136,1187,1057,1060,1135,1258,1430,1425,1195,1008,825,696,669,581,562,668,793,923,1055,1024,882,762,646,524,443,378,311,341,383,392,331,192,158,284,393,337,168,129,145,87,31,-162,-486,-668,-641,-556,-449,-402,-377,-257,-179,-268,-370,-499,-665,-742,-708,-571,-498,-553,-714,-992,-1163,-1098,-1045,-1017,-961,-853,-730,-651,-580,-584,-735,-1006,-1151,-1099,-1045,-1065,-1139,-1274,-1368,-1312,-1218,-1164,-1098,-1019,-956,-961,-949,-955,-993,-976,-928,-844,-745,-703,-676,-633,-658,-761,-823,-719,-597,-516,-390,-328,-388,-476,-559,-559,-407,-267,-281,-288,-267,-322,-339,-297,-307,-238,-111,4,115,218,243,190,186,230,326,418,543,581,462,351,302,237,319,599,828,981,1003,867,721,688,584,474,586,748,812,815,869,856,887,956,905,931,1101,1188,1090,1012,1043,1018,1026,1057,994,959,934,917,838,745,720,749,882,883,792,832,936,985,1000,1048,1045,1066,1091,1068,1022,861,661,554,537,511,456,430,440,534,665,640,636,710,590,311,34,-165,-139,127,368,455,468,375,304,385,345,165,41,-80,-236,-327,-427,-547,-609,-650,-635,-530,-472,-486,-567,-646,-557,-410,-252,-195,-410,-615,-726,-833,-827,-681,-634,-737,-801,-851,-942,-926,-819,-707,-633,-590,-603,-705,-885,-1069,-1155,-1159,-1155,-1215,-1312,-1291,-1189,-1058,-975,-964,-924,-910,-896,-909,-881,-805,-827,-837,-827,-850,-836,-766,-761,-840,-899,-903,-819,-625,-574,-634,-663,-646,-470,-244,-79,-40,-148,-271,-380,-495,-455,-275,-219,-317,-332,-228,-61,129,149,83,68,66,101,155,252,308,245,230,294,441,533,517,547,549,601,752,843,773,667,623,530,460,445,420,461,623,746,787,841,854,819,784,762,811,930,948,919,924,932,951,966,961,942,849,769,794,817,862,871,764,675,682,709,737,849,947,1054,1149,1102,905,701,503,371,350,372,364,418,531,529,518,539,568,615,654,588,384,216,181,192,242,291,171,39,6,66,141,112,102,135,89,-92,-242,-328,-430,-478,-511,-488,-403,-360,-411,-532,-618,-561,-421,-412,-463,-507,-555,-575,-655,-738,-827,-889,-846,-770,-662,-592,-604,-652,-690,-620,-618,-763,-928,-1045,-1078,-1101,-1185,-1261,-1239,-1251,-1101,-883,-830,-777,-754,-726,-650,-631,-696,-833,-846,-694,-588,-619,-796,-937,-920,-806,-621,-487,-491,-515,-432,-356,-344,-368,-321,-175,-128,-121,-120,-172,-204,-138,-101,-79,-42,-72,-29,-27,-9,87,184,273,313,332,337,345,394,458,452,447,532,611,603,449,376,596,810,951,1067,1065,1008,983,932,817,791,767,662,577,610,739,918,978,807,736,819,940,1053,1118,1171,1169,1147,1062,1018,1052,1005,924,860,807,815,880,945,875,813,939,1022,1025,1009,1004,1100,1207,1197,1044,790,537,407,523,638,637,608,555,593,751,863,760,637,562,435,341,149,-20,-40,-51,3,105,108,102,88,91,105,148,229,164,28,-179,-359,-401,-429,-541,-728,-775,-773,-829,-817,-805,-821,-710,-595,-591,-597,-649,-836,-976,-935,-809,-638,-572,-597,-730,-806,-718,-705,-809,-886,-886,-925,-998,-1117,-1229,-1198,-1157,-1253,-1396,-1500,-1490,-1359,-1180,-1064,-919,-724,-699,-719,-615,-611,-770,-891,-927,-835,-717,-699,-799,-808,-737,-702,-650,-646,-626,-569,-551,-564,-542,-452,-364,-260,-179,-171,-246,-306,-236,-229,-216,-204,-241,-105,60,154,211,284,326,220,226,353,404,402,363,274,166,253,392,351,373,586,717,839,1083,1111,1058,1083,971,905,997,993,975,1054,1054,1007,1054,1065,1016,968,969,972,984,1040,1161,1268,1306,1298,1175,991,902,821,774,960,1177,1220,1163,1047,1035,1146,1195,1100,1003,1066,1109,1174,1163,1012,818,639,540,507,554,579,593,651,672,715,787,876,889,746,657,581,449,294,115,70,115,223,286,214,97,26,-45,-93,-52,-42,-140,-195,-191,-258,-357,-419,-486,-564,-592,-565,-488,-422,-406,-358,-316,-435,-622,-656,-640,-661,-707,-837,-893,-776,-688,-738,-817,-788,-728,-746,-801,-865,-952,-966,-969,-1066,-1087,-1083]
    padding_data = np.array(pad_list, dtype=np.int16)
    pad_num = len(padding_data)
    for i in range(0,data_len,interval):
        raw_chunk_data = np.concatenate([padding_data, data[i:min(i+interval, data_len)]])
        # chunk_data = data[i:min(i+interval, data_len)]
        # print(",".join(list(map(str, list(chunk_data[:pad_num]) ))) )
        processed_chunk_data = logmmse.logmmse(data=raw_chunk_data, sampling_rate=sample_rate)
        processed_data.append(processed_chunk_data[pad_num:])
        print(len(raw_chunk_data), len(processed_data[-1]) )
        # displayWaveform([raw_chunk_data, processed_chunk_data])

    # processed_data = logmmse.logmmse(data=data, sampling_rate=sample_rate)
    processed_data = np.concatenate(processed_data)
    print(len(processed_data))
    displayWaveform([data, processed_data])
    sf.write("/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_10s_logmmse.wav", processed_data, sample_rate)



    # n_std_thresh': 1.5755461792807288, 'prop_decrease': 0.2316055424526226
    # n_std_thresh = trial.suggest_uniform('n_std_thresh', 0., 2.)
    # prop_decrease = trial.suggest_uniform('prop_decrease', 0., 2.)
    # print("params ", n_std_thresh, prop_decrease)
    # data = data.astype(np.float32)
    # data = removeNoise(audio_clip=data, noise_clip=data,
    #                    n_std_thresh=n_std_thresh, prop_decrease=prop_decrease,
    #                    verbose=True, visual=False)
    # data = data.astype(np.int16)
    # print("data ", data[10000:10100])

    def get_valid_chunk(data, chunk_period = 0.1, invalid_period_thr=0.7):

        chunk_len = int(chunk_period * sample_rate)
        data = np.concatenate([data, np.zeros(int(chunk_len-data_len%chunk_len), dtype=np.float16)])
        is_valid = np.mean( (data**2).reshape((-1, chunk_len)), axis=1)
        is_valid = (is_valid>is_valid.mean()*0.15).reshape(-1)
        print("data len ", len(data))
        print("is_valid len ", len(is_valid))

        displayWaveform([data, is_valid])

        # data = data.astype(np.int16)

        cur_invalid_cnt = 0
        cur_valid = False
        last_idx = 0
        chunk_list = []
        for i in range(len(is_valid)):
            if not is_valid[i]:
                cur_invalid_cnt += 1
                if cur_invalid_cnt >= invalid_period_thr//chunk_period:
                    if cur_valid:
                        chunk_list.append(np.array([last_idx, i+1])*chunk_len)
                        cur_valid = False
                    last_idx = i
            else:
                cur_valid = True
                cur_invalid_cnt = 0


        # print(is_valid)
        print(chunk_list)
        return chunk_list

    # chunk_list = get_valid_chunk(data)
    chunk_list = [np.array([0,data_len])]
    total_text = ""
    for chunk in chunk_list:
            chunk_wav = data[chunk[0]: chunk[1]].tobytes()
            total_text += asyncio.run(ws_rec(chunk_wav, ws_uri))['text']
            # print(chunk, total_text)
            if total_text:
                sys.stdout.write('\r' + total_text)
                sys.stdout.flush()

    total_text = total_text.replace('</context>', '').replace('<context>', '')
    print("\r\n" + total_text)

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
    edit_dis = minDistance(label, total_text)
    print("edit_dis ", edit_dis)
    return edit_dis




def request_server_continue_decord(input_file):

    WS_START = json.dumps({
        'signal': 'start',
        'nbest': 1,
        'continuous_decoding': True,
    })
    WS_END = json.dumps({
        'signal': 'end'
    })
    ws_uri = "ws://127.0.0.1:10086"

    async def ws_rec(data, ws_uri):

        texts = []

        begin = time.time()
        conn = await websockets.connect(ws_uri, ping_timeout=200)
        print("connected ")
        # step 1: send start
        await conn.send(WS_START)
        ret = await conn.recv()
        print(ret)
        # step 2: send audio data

        interval = int(2 * 16000)
        print("data len", len(data))
        for i in range(0, len(data), interval):
            chunk_wav = data[i: min(i + interval, len(data))].tobytes()
            await conn.send(chunk_wav)
        # step 3: send end
        await conn.send(WS_END)
        # step 4: receive result

        while 1:
            ret = await conn.recv()
            ret = json.loads(ret)
            # print(ret)
            if ret['type'] == 'final_result':
                nbest = json.loads(ret['nbest'])
                text = nbest[0]['sentence']
                texts.append(text)
                if texts:
                    sys.stdout.write('\r' + "".join(texts))
                    sys.stdout.flush()
            elif ret['type'] == 'speech_end':
                break
        # step 5: close
        try:
            await conn.close()
        except Exception as e:
            # this except has no effect, just log as debug
            # it seems the server does not send close info, maybe
            # print(e)
            pass
        time_cost = time.time() - begin
        return {
            'text': ''.join(texts),
            'time': time_cost,
        }

    data, sr = sf.read(input_file, dtype='int16')
    assert sr == 16000
    duration = (len(data)) / 16000
    print(duration, type(data))

    result = asyncio.run(ws_rec(data, ws_uri))
    print(result)

def realtime_webserver_decord(record_second=10):

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = RATE*1
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    WS_START = json.dumps({
        'signal': 'start',
        'nbest': 1,
        'continuous_decoding': True,
    })
    WS_END = json.dumps({
        'signal': 'end'
    })
    ws_uri = "ws://127.0.0.1:10086"

    async def ws_rec(data, ws_uri):

        begin = time.time()
        conn = await websockets.connect(ws_uri, ping_timeout=200)
        # print("connected ")
        # step 1: send start
        await conn.send(WS_START)
        ret = await conn.recv()
        # print(ret)

        texts = []

        await conn.send(data)

        await conn.send(WS_END)

        # print("send over")
        while 1:
            ret = await conn.recv()
            ret = json.loads(ret)
            # print(ret)
            if ret['type'] == 'final_result':
                nbest = json.loads(ret['nbest'])
                text = nbest[0]['sentence']
                texts.append(text)

            elif ret['type'] == 'speech_end':
                break

        try:
            await conn.close()
        except Exception as e:
            pass
        time_cost = time.time() - begin
        return {
            'text': "".join(texts),
            'time': time_cost,
        }

    total_text = ""
    total_chunk = int(RATE / CHUNK * record_second)
    for i in range(0, total_chunk + 1):
        data = stream.read(CHUNK)
        total_text += asyncio.run(ws_rec(data, ws_uri))['text']
        if total_text:
            sys.stdout.write('\r' + "".join(total_text))
            sys.stdout.flush()
    print(total_text)


    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

def realtime_webserver_decord_continue(record_second=10):

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = RATE*2
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    WS_START = json.dumps({
        'signal': 'start',
        'nbest': 1,
        'continuous_decoding': False,
    })
    WS_END = json.dumps({
        'signal': 'end'
    })
    ws_uri = "ws://127.0.0.1:10086"

    async def ws_rec(stream, ws_uri):

        begin = time.time()
        conn = await websockets.connect(ws_uri, ping_timeout=200)
        # print("connected ")
        # step 1: send start
        await conn.send(WS_START)
        ret = await conn.recv()
        # print(ret)

        texts = []

        total_chunk = int(RATE / CHUNK * record_second)
        for i in range(0, total_chunk + 1):
            print(i)
            data = stream.read(CHUNK)
            await conn.send(data)

            ret = await conn.recv()
            # ret = json.loads(ret)
            # print(ret)
            # if ret['type'] == 'final_result':
            #     nbest = json.loads(ret['nbest'])
            #     text = nbest[0]['sentence']
            #     texts.append(text)



        await conn.send(WS_END)

        print("send over")
        while 1:
            ret = await conn.recv()
            ret = json.loads(ret)
            print(ret)
            if ret['type'] == 'final_result':
                nbest = json.loads(ret['nbest'])
                text = nbest[0]['sentence']
                texts.append(text)

                if texts:
                    sys.stdout.write('\r' + ",".join(texts))
                    sys.stdout.flush()
            elif ret['type'] == 'speech_end':
                break




        try:
            await conn.close()
        except Exception as e:
            pass
        time_cost = time.time() - begin
        return {
            'text': ",".join(texts),
            'time': time_cost,
        }

    total_text = asyncio.run(ws_rec(stream, ws_uri))
    print(total_text)


    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()





# stream_decord(wav_file_10s)

request_server_decord()
# realtime_webserver_decord()

def optuna_study(obj):
    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials=200)
    print(study.best_params)
    print(study.best_value)

    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

