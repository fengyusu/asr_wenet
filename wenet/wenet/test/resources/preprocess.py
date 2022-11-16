# !/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import json
import ast
import asyncio
from collections import deque
from tqdm import tqdm
import datetime
from datetime import timedelta as td
import time


import websockets
import soundfile as sf
import librosa
import pyaudio
import wave
from scipy.io import wavfile
import scipy.signal
import struct as st

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import wenetruntime as wenet


wav_file = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output.wav"
wav_file_10s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_10s.wav"
wav_file_60s = "/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_60s.wav"
wav_file_myhome = r"/media/sfy/File/我的视频/myhome.mkv.wav"


def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.57,
    prop_decrease=0.23,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """

    def _stft(y, n_fft, hop_length, win_length):
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _istft(y, hop_length, win_length):
        return librosa.istft(y, hop_length, win_length)

    def _amp_to_db(x):
        return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

    def _db_to_amp(x, ):
        return librosa.core.db_to_amplitude(x, ref=1.0)

    def plot_spectrogram(signal, title):
        fig, ax = plt.subplots(figsize=(20, 4))
        cax = ax.matshow(
            signal,
            origin="lower",
            aspect="auto",
            cmap=plt.cm.seismic,
            vmin=-1 * np.max(np.abs(signal)),
            vmax=np.max(np.abs(signal)),
        )
        fig.colorbar(cax)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_statistics_and_filter(mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter):
        fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
        plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
        plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
        plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
        ax[0].set_title("Threshold for mask")
        ax[0].legend()
        cax = ax[1].matshow(smoothing_filter, origin="lower")
        fig.colorbar(cax)
        ax[1].set_title("Filter for smoothing Mask")
        plt.show()


    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal

def noise_reduce(input_file=wav_file_60s):
    data, sr = sf.read(input_file, dtype='int16')
    data = data.astype(np.float32)
    data_len = len(data)
    time = np.arange(0, len(data)) * (1.0 / sr)

    # noise_len = 60  # seconds
    # noise = band_limited_noise(min_freq=4000, max_freq=12000, samples=len(data), samplerate=sr) * 10
    # noise_clip = noise[:sr * noise_len]

    cands = [512]
    n = len(cands) + 1
    plt.subplot(n, 1, 1)
    plt.plot(time, data)
    for i in range(n-1):
        output = removeNoise(audio_clip=data, noise_clip=data, hop_length=cands[i], verbose=True, visual=False)

        plt.subplot(n, 1, i+2)
        plt.plot(output)

    plt.show()

    output = output.astype(np.int16)
    sf.write(file="/media/sfy/Study/graduation/asr_wenet/wenet/wenet/test/test_data/output_60s_noise_reduce.wav", data=output, samplerate=sr)


noise_reduce()

def VAD(signal, fs):
    def ShortTimeEnergy(signal, windowLength, step):
        """
        计算短时能量
        Parameters
        ----------
        signal : 原始信号.
        windowLength : 帧长.
        step : 帧移.

        Returns
        -------
        E : 每一帧的能量.
        """
        signal = signal / np.max(signal)  # 归一化
        curPos = 0
        L = len(signal)
        numOfFrames = np.asarray(np.floor((L - windowLength) / step) + 1, dtype=int)
        E = np.zeros((numOfFrames, 1))
        for i in range(numOfFrames):
            window = signal[int(curPos):int(curPos + windowLength - 1)];
            E[i] = (1 / (windowLength)) * np.sum(np.abs(window ** 2));
            curPos = curPos + step;
        return E

    def SpectralCentroid(signal, windowLength, step, fs):
        """
        计算谱质心
        Parameters
        ----------
        signal : 原始信号.
        windowLength : 帧长.
        step : 帧移.
        fs : 采样率.

        Returns
        -------
        C : 每一帧的谱质心.
        """
        signal = signal / np.max(signal)  # 归一化
        curPos = 0
        L = len(signal)
        numOfFrames = np.asarray(np.floor((L - windowLength) / step) + 1, dtype=int)
        H = np.hamming(windowLength)
        m = ((fs / (2 * windowLength)) * np.arange(1, windowLength, 1)).T
        C = np.zeros((numOfFrames, 1))
        for i in range(numOfFrames):
            window = H * (signal[int(curPos): int(curPos + windowLength)])
            FFT = np.abs(np.fft.fft(window, 2 * int(windowLength)))
            FFT = FFT[1: windowLength]
            FFT = FFT / np.max(FFT)
            C[i] = np.sum(m * FFT) / np.sum(FFT)
            if np.sum(window ** 2) < 0.010:
                C[i] = 0.0
            curPos = curPos + step;
        C = C / (fs / 2)
        return C

    def findMaxima(f, step):
        """
        寻找局部最大值
        Parameters
        ----------
        f : 输入序列.
        step : 搜寻窗长.

        Returns
        -------
        Maxima : 最大值索引 最大值
        countMaxima : 最大值的数量
        """
        ## STEP 1: 寻找最大值
        countMaxima = 0
        Maxima = []
        for i in range(len(f) - step - 1):  # 对于序列中的每一个元素:
            if i >= step:
                if (np.mean(f[i - step: i]) < f[i]) and (np.mean(f[i + 1: i + step + 1]) < f[i]):
                    # IF the current element is larger than its neighbors (2*step window)
                    # --> keep maximum:
                    countMaxima = countMaxima + 1
                    Maxima.append([i, f[i]])
            else:
                if (np.mean(f[0: i + 1]) <= f[i]) and (np.mean(f[i + 1: i + step + 1]) < f[i]):
                    # IF the current element is larger than its neighbors (2*step window)
                    # --> keep maximum:
                    countMaxima = countMaxima + 1
                    Maxima.append([i, f[i]])

        ## STEP 2: 对最大值进行进一步处理
        MaximaNew = []
        countNewMaxima = 0
        i = 0
        while i < countMaxima:
            # get current maximum:

            curMaxima = Maxima[i][0]
            curMavVal = Maxima[i][1]

            tempMax = [Maxima[i][0]]
            tempVals = [Maxima[i][1]]
            i = i + 1

            # search for "neighbourh maxima":
            while (i < countMaxima) and (Maxima[i][0] - tempMax[len(tempMax) - 1] < step / 2):
                tempMax.append(Maxima[i][0])
                tempVals.append(Maxima[i][1])
                i = i + 1

            MM = np.max(tempVals)
            MI = np.argmax(tempVals)
            if MM > 0.02 * np.mean(f):  # if the current maximum is "large" enough:
                # keep the maximum of all maxima in the region:
                MaximaNew.append([tempMax[MI], f[tempMax[MI]]])
                countNewMaxima = countNewMaxima + 1  # add maxima
        Maxima = MaximaNew
        countMaxima = countNewMaxima

        return Maxima, countMaxima

    win = 0.05
    step = 0.05
    Eor = ShortTimeEnergy(signal, int(win * fs), int(step * fs));
    Cor = SpectralCentroid(signal, int(win * fs), int(step * fs), fs);
    E = scipy.signal.medfilt(Eor[:, 0], 5)
    E = scipy.signal.medfilt(E, 5)
    C = scipy.signal.medfilt(Cor[:, 0], 5)
    C = scipy.signal.medfilt(C, 5)

    E_mean = np.mean(E);
    Z_mean = np.mean(C);
    Weight = 100  # 阈值估计的参数
    # 寻找短时能量的阈值
    Hist = np.histogram(E, bins=10)  # 计算直方图
    HistE = Hist[0]
    X_E = Hist[1]
    MaximaE, countMaximaE = findMaxima(HistE, 3)  # 寻找直方图的局部最大值
    if len(MaximaE) >= 2:  # 如果找到了两个以上局部最大值
        T_E = (Weight * X_E[MaximaE[0][0]] + X_E[MaximaE[1][0]]) / (Weight + 1)
    else:
        T_E = E_mean / 2

    # 寻找谱质心的阈值
    Hist = np.histogram(C, bins=10)
    HistC = Hist[0]
    X_C = Hist[1]
    MaximaC, countMaximaC = findMaxima(HistC, 3)
    if len(MaximaC) >= 2:
        T_C = (Weight * X_C[MaximaC[0][0]] + X_C[MaximaC[1][0]]) / (Weight + 1)
    else:
        T_C = Z_mean / 2

    # 阈值判断
    Flags1 = (E >= T_E)
    Flags2 = (C >= T_C)
    flags = np.array(Flags1 & Flags2, dtype=int)

    ## 提取语音片段
    count = 1
    segments = []
    while count < len(flags):  # 当还有未处理的帧时
        # 初始化
        curX = []
        countTemp = 1
        while ((flags[count - 1] == 1) and (count < len(flags))):
            if countTemp == 1:  # 如果是该语音段的第一帧
                Limit1 = np.round((count - 1) * step * fs) + 1  # 设置该语音段的开始边界
                if Limit1 < 1:
                    Limit1 = 1
            count = count + 1  # 计数器加一
            countTemp = countTemp + 1  # 当前语音段的计数器加一

        if countTemp > 1:  # 如果当前循环中有语音段
            Limit2 = np.round((count - 1) * step * fs)  # 设置该语音段的结束边界
            if Limit2 > len(signal):
                Limit2 = len(signal)
            # 将该语音段的首尾位置加入到segments的最后一行
            segments.append([int(Limit1), int(Limit2)])
        count = count + 1

    # 合并重叠的语音段
    for i in range(len(segments) - 1):  # 对每一个语音段进行处理
        if segments[i][1] >= segments[i + 1][0]:
            segments[i][1] = segments[i + 1][1]
            segments[i + 1, :] = []
            i = 1

    return segments

def vad_test(input_file=wav_file_60s):
    CHUNK = 1600
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 通道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = 3  # 时长

    sample_rate = 16000
    data, sr = sf.read(input_file, dtype='int16')
    assert sr == sample_rate
    data_len = len(data)
    duration = data_len / sample_rate
    print(duration, type(data))

    signal = data
    segments = VAD(signal, RATE)  # 端点检测
    print(segments)

    index = 0
    for seg in segments:
        if index < seg[0]:
            x = np.linspace(index, seg[0], seg[0] - index, endpoint=True, dtype=int)
            y = signal[index:seg[0]]
        plt.plot(x, y, 'g', alpha=1)
        x = np.linspace(seg[0], seg[1], seg[1] - seg[0], endpoint=True, dtype=int)
        y = signal[seg[0]:seg[1]]
        plt.plot(x, y, 'r', alpha=1)
        index = seg[1]
    x = np.linspace(index, len(signal), len(signal) - index, endpoint=True, dtype=int)
    y = signal[index:len(signal)]
    plt.plot(x, y, 'g', alpha=1)
    plt.ylim((-32768, 32767))
    plt.show()

# vad_test()

def vad_realtime_record():
    CHUNK = 1600
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 通道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = 3  # 时长
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []  # 音频缓存
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if (len(frames) > RECORD_SECONDS * RATE / CHUNK):
            del frames[0]
        datas = b''
        for i in range(len(frames)):
            datas = datas + frames[i]
        if len(datas) == RECORD_SECONDS * RATE * 2:
            fmt = "<" + str(RECORD_SECONDS * RATE) + "h"
            signal = np.array(st.unpack(fmt, bytes(datas)))  # 字节流转换为int16数组
            segments = VAD(signal, RATE)  # 端点检测
            # 可视化
            index = 0
            for seg in segments:
                if index < seg[0]:
                    x = np.linspace(index, seg[0], seg[0] - index, endpoint=True, dtype=int)
                    y = signal[index:seg[0]]
                    plt.plot(x, y, 'g', alpha=1)
                x = np.linspace(seg[0], seg[1], seg[1] - seg[0], endpoint=True, dtype=int)
                y = signal[seg[0]:seg[1]]
                plt.plot(x, y, 'r', alpha=1)
                index = seg[1]
            x = np.linspace(index, len(signal), len(signal) - index, endpoint=True, dtype=int)
            y = signal[index:len(signal)]
            plt.plot(x, y, 'g', alpha=1)
            plt.ylim((-32768, 32767))
            plt.show()

import pycorrector


str = "我与父亲不相见已二年余了我对付能忘记的是他的背影那年冬天如如死了父亲的差始也交谢了中是祸不该行的日子我从北京到徐州打算跟着父亲奔乡回家到徐州按照父亲看见满月狼藉的东西又想起祖母不拘簌簌地流下眼泪父亲说事已如此不易难过好在燕无绝人之路回家变卖贬质父亲怀了沟空又借钱办了方事这些日子加工光景很是残废一半为了伤事一半为了父亲赋闲丧事完毕父亲要到南京谋事我也要回北京念书我们便同行怕南京时有朋友约去逛街勾留了一日第二日上午"

# corrected_sent, detail = pycorrector.correct('祸不行')
# corrected_sent, detail = pycorrector.correct(str)
# print(corrected_sent, detail)







