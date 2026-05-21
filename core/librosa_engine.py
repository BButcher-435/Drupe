import librosa
import numpy as np
import sounddevice as sd  
from collections import deque # Hareketli ortalamalar (moving average) için performanslı liste
import threading # Sesi dinlerken sistemi/arayüzü dondurmamak için
import queue # Thread'ler arası güvenli veri aktarımı için
import time

# --- SES YAKALAMA SABİTLERİ ---
SAMPLE_RATE = 44100  # Standart ses kalitesi: Equalizer APO ile aynı olmalı = 44.1 kHz 
CHANNELS = 2         # Stereo ses için iki kanal
BUFFER_DURATION = 5  # Librosa'ya bir seferde kaç saniyelik ses gönderilecek? 
CHUNK_SIZE = int(SAMPLE_RATE * BUFFER_DURATION) # Tampon boyutu, büyük olması işlemci yükünü azaltır.

# - EQ BANTLARI -
FREQUENCIES = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000]

def calculate_preamp(target_bands): # preamp hesaplama 
    max_gain = max(target_bands.values())
    return round(min(0.0, -max_gain), 2)

def apply_q_factor(bands_dict, spread_factor=0.3): # q faktörü 
    smoothed_q_bands = bands_dict.copy()
    for i in range(len(FREQUENCIES)):
        current_freq = FREQUENCIES[i]
        if abs(bands_dict[current_freq]) > 0.5: 
            if i > 0: smoothed_q_bands[FREQUENCIES[i-1]] += bands_dict[current_freq] * spread_factor
            if i < len(FREQUENCIES) - 1: smoothed_q_bands[FREQUENCIES[i+1]] += bands_dict[current_freq] * spread_factor
    for freq in FREQUENCIES:
        smoothed_q_bands[freq] = max(-12.0, min(12.0, smoothed_q_bands[freq]))
    return smoothed_q_bands

def apply_smoothing(current_bands, target_bands, alpha_attack=0.2, alpha_release=0.05): # smooth gecis
    smoothed_bands = {}
    for freq in FREQUENCIES:
        eski = current_bands.get(freq, 0.0)
        hedef = target_bands.get(freq, 0.0)
        yeni = (alpha_attack * hedef) + ((1.0 - alpha_attack) * eski) if hedef > eski else (alpha_release * hedef) + ((1.0 - alpha_release) * eski)
        smoothed_bands[freq] = round(yeni, 2)
    return smoothed_bands

def apply_librosa_tweaks(base_bands, librosa_metrics): # librosa metrikleri 
    tweaked_bands = base_bands.copy()
    rms = librosa_metrics.get("rms", 0.05)
    zcr = librosa_metrics.get("zcr", 0.0)
    centroid = librosa_metrics.get("centroid", 2000)
    flux = librosa_metrics.get("flux", 0.0)
    
    intensity_factor = min(1.0, max(0.1, rms * 10)) 
    
    if zcr > 0.08: tweaked_bands[2500] += (1.0 * intensity_factor) 
    if flux > 1.5: 
        tweaked_bands[63] += (2.5 * intensity_factor)
        tweaked_bands[100] += (1.5 * intensity_factor)
    if centroid > 4000:
        tweaked_bands[4000] -= (1.0 * intensity_factor)
        tweaked_bands[6300] -= (1.5 * intensity_factor)
        tweaked_bands[10000] -= (2.0 * intensity_factor)
        
    for freq in FREQUENCIES:
        tweaked_bands[freq] = round(max(-12.0, min(12.0, tweaked_bands[freq])), 2)
        
    return apply_q_factor(tweaked_bands, spread_factor=0.35)