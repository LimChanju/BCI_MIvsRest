##################################################################
# 주파수 -> time-frequency map
# 이로써 CNN이 아닌 2D frame 기반 모델 입력으로 사용할 수 있음.
##################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# X: (trials, 1, times), fs: 샘플링 주파수 (예: 250 Hz)
fs = 250
f, t, Zxx = stft(X[0, 0, :], fs=fs, nperseg=128, noverlap=64)
spectrogram = np.abs(Zxx)

plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, spectrogram, shading='gouraud')
plt.title('EEG Spectrogram (C3)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.savefig("./spectrogram_C3.png", dpi=300)


################################################################
# Mel-spectrogram / MFCC 변환 방식
################################################################
import librosa
import librosa.display

# mel-spectrogram
signal = X[0, 0, :]
fs = 250
mel_spec = librosa.feature.melspectrogram(
    y=signal, sr=fs, n_fft=256, hop_length=128, n_mels=40, fmax=50
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# mfcc
mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13, n_fft=256, hop_length=128)


plt.figure(figsize=(8, 4))
librosa.display.specshow(mel_spec_db, sr=fs, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram of EEG (C3)')
plt.tight_layout()
plt.show()
plt.savefig("./mel_spectrogram_C3.png", dpi=300)

#################################################################
# 모델 아이디어
# 1. EEG-MelNet (EEG -> Mel-spectrogram -> CNN (Resnet 기반))
# 2. EEG-MFCCNet (EEG -> MFCC -> BiLSTM후 FC) (주파수 계수 간 순서 학습)
# 3. EEG-TFTransformer (EEG -> SFTF -> Vision Transformer) (ERD/DRS 패턴 위치 불변 학습)
#################################################################

#1, EEG-MelNet
import torch
import torch.nn as nn

class EEG_MelNet(nn.Module):
    def __init__(self, n_mels=40, n_frames=64, n_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (n_mels//2) * (n_frames//2), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):  # x: (B, 1, n_mels, n_frames)
        x = self.conv(x)
        return self.fc(x)

# 이제 다음 중 하나로 확장할 수도 있어 👇

μ-band(8–12Hz) / β-band(13–30Hz) 각각 따로 mel 변환해서 concat

Grad-CAM으로 모델이 주목하는 시간-주파수 영역 시각화

EEGNet 구조로 교체해서 mel feature를 입력으로 넣기

# EEG-NeuroMelVit 결과 : 56%