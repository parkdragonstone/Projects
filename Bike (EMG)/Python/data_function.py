from scipy.signal import welch
from scipy.fftpack import fft
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# 기본 폰트를 Arial로 설정
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


def power_fft(df, emg_sr):
    FFT = {
        'freq' : {},
        'power' : {}
    }
    for c in ['RF','VL','BF','GM']:
        x = np.array(df[c])
        N = len(x)
        X = fft(x, N)
        freq = np.fft.fftfreq(N, 1/emg_sr)
        half_N = N//2
        freq = freq[:half_N]
        X = X[:half_N]
        power = np.abs(X)**2 / N
        FFT['freq'][c] = freq
        FFT['power'][c] = power
 
    return FFT

def power_fft_plot(FFT, trial):
    fig, ax = plt.subplots(1,4, figsize = (15,4))
    for idx,m in enumerate(FFT['freq']):
        # 결과 시각화
        freq = FFT['freq'][m]
        power = FFT['power'][m]
        # 누적 전력 스펙트럼 밀도 계산
        median_frequency = median_freq(freq, power)
        
        ax[idx].plot(freq, power, color = 'b')
        ax[idx].axvline(median_frequency, color='r', linestyle='--', label=f'MF: {median_frequency:.2f} Hz')
        ax[idx].set_title(f'{m}', size=15)
        ax[idx].set_xlabel('Frequency [Hz]', size=15)
        ax[idx].grid(True)
        ax[idx].legend(loc='upper right')

    ax[0].set_ylabel('Power [V²]', size=15)
    plt.suptitle(trial, size = 15)  
    plt.tight_layout()
    plt.show()

def welch_fft(df, emg_sr, N):
    FFT = {
        'freq' : {},
        'power' : {},
    }
    for c in ['RF','VL','BF','GM']:
        x = np.array(df[c])
        freq, power = welch(x, emg_sr, window='hann', nperseg=N)
        FFT['freq'][c] = freq
        FFT['power'][c] = power

    return FFT

def power_welch_plot(FFT, trial):
    fig, ax = plt.subplots(1,4, figsize = (15,4))
    for idx,m in enumerate(FFT['freq']):
        # 결과 시각화
        freq = FFT['freq'][m]
        power = FFT['power'][m]
        # 누적 전력 스펙트럼 밀도 계산
        median_frequency = median_freq(freq, power)
        
        ax[idx].plot(freq, power, color = 'b')
        ax[idx].axvline(median_frequency, color='r', linestyle='--', label=f'Central Frequency: {median_frequency:.2f} Hz')
        ax[idx].set_title(f'{m}', size=15)
        ax[idx].set_xlabel('Frequency [Hz]', size=15)
        ax[idx].grid(True)
        ax[idx].legend(loc='upper right')

    ax[0].set_ylabel('PSD [V²/Hz]', size=15)
    plt.suptitle(trial, size = 15)  
    plt.tight_layout()
    plt.show()

def median_freq(freq, power):
    # 누적 전력 스펙트럼 밀도 계산
    cumulative_power = np.cumsum(power)
    total_power = cumulative_power[-1]
    half_power_point = total_power / 2

    # 중앙 주파수 계산 (면적의 절반 지점)
    median_frequency_index = np.where(cumulative_power >= half_power_point)[0][0]
    median_frequency = freq[median_frequency_index]

    return median_frequency

class filtering():
    def lowpassfilter(x, fs ,cutoff, order):
        b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        # 필터 적용
        y = filtfilt(b, a, x)
        return y

    def bandpassfilter(x, fs, lowcut, highcut, order):
        b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
        # 필터 적용
        y = filtfilt(b, a, x)
        return y

    def notchfilter(x, fs, notch_freq):    
        quality_factor = 30  # 품질 계수
        b, a = iirnotch(notch_freq / (0.5 * fs), quality_factor)
        # 필터 적용
        y = filtfilt(b, a, x)
        return y
