import os
import sys

# 1. cffi ve ctypes C-Motoru için kesin PATH zorlaması (ole32 Hatasını Çözen Kısım)
os.environ["PATH"] = r"C:\Windows\System32;" + os.environ.get("PATH", "")

# 2. Python 3.8 ve üzeri için DLL kuralı
if os.name == 'nt' and sys.version_info >= (3, 8):
    try:
        os.add_dll_directory(r"C:\Windows\System32")
    except Exception:
        pass

# Gereksiz sarı uyarıları sustur
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundcard as sc
import librosa
import threading
import queue
import pythoncom
import time
from .eq_controller import update_eq_from_audio_features

# ─────────────────────────────────────────────────────────────────
# SES ANALİZİ SABİTLERİ
# ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050
BUFFER_SECONDS = 5
CHUNK_SECONDS  = 0.40
CHUNK_FRAMES   = int(SAMPLE_RATE * CHUNK_SECONDS)
BUFFER_FRAMES  = int(SAMPLE_RATE * BUFFER_SECONDS)
EMA_ALPHA      = 0.35
N_FFT          = 2048
HOP_LENGTH     = 512

BANDS = [
    ("b01", "20–40 Hz", 20, 40), ("b02", "40–80 Hz", 40, 80),
    ("b03", "80–120 Hz", 80, 120), ("b04", "120–200 Hz", 120, 200),
    ("b05", "200–315 Hz", 200, 315), ("b06", "315–500 Hz", 315, 500),
    ("b07", "500–800 Hz", 500, 800), ("b08", "800 Hz–1.2k", 800, 1200),
    ("b09", "1.2–2k Hz", 1200, 2000), ("b10", "2–3.15k Hz", 2000, 3150),
    ("b11", "3.15–5k Hz", 3150, 5000), ("b12", "5–8k Hz", 5000, 8000),
    ("b13", "8–10k Hz", 8000, 10000), ("b14", "10–13k Hz", 10000, 13000),
    ("b15", "13–16k Hz", 13000, 16000),
]
BAND_KEYS = [b[0] for b in BANDS]

def get_audio_state() -> dict:
    return {
        "result_queue": queue.Queue(maxsize=3),
        "stop_event":   threading.Event(),
        "ring_buffer":  np.zeros(BUFFER_FRAMES, dtype=np.float32),
        "ring_lock":    threading.Lock(),
        "ema":          {},
        "worker":       None,
        "chunk_buffer": [],
        "period_start_time": time.time(),
        "aggregated_data": [],
        "agg_lock": threading.Lock(),
    }

def _ema(store: dict, key: str, value: float) -> float:
    prev = store.get(key, value)
    v    = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * prev
    store[key] = v
    return float(v)

def extract_features(y: np.ndarray, ema_store: dict) -> dict:
    KEYS = ["tempo", "energy_rms", "loudness_a", "danceability", "zcr", "spectral_flux", "spectral_centroid", "spectral_rolloff", *BAND_KEYS]
    if float(np.max(np.abs(y))) < 1e-5:
        return {k: ema_store.get(k, 0.0) for k in KEYS}

    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) < 1024:
        y_trimmed = y
    y = y_trimmed
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_power = np.abs(D) ** 2
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

    energy_rms = _ema(ema_store, "energy_rms", float(np.mean(librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0])))
    aw_db = librosa.perceptual_weighting(S_power + 1e-10, freqs, ref=1.0)
    loudness_a = _ema(ema_store, "loudness_a", float(np.mean(aw_db)))

    raw_tempo, _ = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    t = float(np.atleast_1d(raw_tempo)[0])
    if t > 0:
        while t < 70:  t *= 2
        while t > 180: t /= 2
    tempo = _ema(ema_store, "tempo", t)

    onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units="time")
    if len(onset_times) >= 4:
        ioi = np.diff(onset_times)
        dance = float(np.clip(1.0 - float(np.std(ioi) / (np.mean(ioi) + 1e-9)), 0.0, 1.0))
    else:
        dance = 0.0
    danceability = _ema(ema_store, "danceability", dance)

    zcr = _ema(ema_store, "zcr", float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH))))
    flux = _ema(ema_store, "spectral_flux", float(np.mean(onset_env)))
    spectral_centroid = _ema(ema_store, "spectral_centroid", float(np.mean(librosa.feature.spectral_centroid(S=np.abs(D), sr=SAMPLE_RATE, n_fft=N_FFT))))
    spectral_rolloff = _ema(ema_store, "spectral_rolloff", float(np.mean(librosa.feature.spectral_rolloff(S=np.abs(D), sr=SAMPLE_RATE, n_fft=N_FFT, roll_percent=0.85))))

    mfcc_mean = np.mean(librosa.feature.mfcc(S=librosa.power_to_db(S_power, ref=np.max), sr=SAMPLE_RATE, n_mfcc=20), axis=1)
    mfcc_dict = {f"mfcc_{i+1}": _ema(ema_store, f"mfcc_{i+1}", float(val)) for i, val in enumerate(mfcc_mean)}

    total_pwr = float(S_power.mean()) + 1e-10
    band_out = {}
    for key, _, f_lo, f_hi in BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_out[key] = _ema(ema_store, key, float(S_power[mask].mean() / total_pwr) if mask.any() else 0.0)

    return {"tempo": tempo, "energy_rms": energy_rms, "loudness_a": loudness_a, "danceability": danceability, "zcr": zcr, "spectral_flux": flux, "spectral_centroid": spectral_centroid, "spectral_rolloff": spectral_rolloff, **band_out, **mfcc_dict}

def average_20_features(state: dict, feats: dict) -> dict:
    with state["agg_lock"]:
        state["chunk_buffer"].append(feats)
        current_time = time.time()
        elapsed = current_time - state["period_start_time"]

        if elapsed >= 20.0:
            avg_feats = None
            if state["chunk_buffer"]:
                avg_feats = {}
                feature_keys = state["chunk_buffer"][0].keys()

                for key in feature_keys:
                    values = [chunk[key] for chunk in state["chunk_buffer"]]
                    avg_feats[key] = float(np.mean(values))

                avg_feats["period_timestamp"] = current_time
                avg_feats["chunk_count"] = len(state["chunk_buffer"])

                state["aggregated_data"].append(avg_feats)

            state["chunk_buffer"] = []
            state["period_start_time"] = current_time
            return avg_feats

    return None

def audio_worker(state: dict) -> None:
    pythoncom.CoInitialize()
    rq, stop, lock, ema_st = state["result_queue"], state["stop_event"], state["ring_lock"], state["ema"]
    try:
        speaker = sc.default_speaker()
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
            while not stop.is_set():
                chunk = recorder.record(numframes=CHUNK_FRAMES)
                mono = chunk.mean(axis=1).astype(np.float32)
                with lock:
                    state["ring_buffer"] = np.roll(state["ring_buffer"], -len(mono))
                    state["ring_buffer"][-len(mono):] = mono
                    y_snap = state["ring_buffer"].copy()
                feats = extract_features(y_snap, ema_st)
                avg_result = average_20_features(state, feats)
                if avg_result is not None:
                    update_eq_from_audio_features(avg_result)
                if rq.full():
                    try: rq.get_nowait()
                    except queue.Empty: pass
                try: rq.put_nowait(feats)
                except queue.Full: pass
    finally:
        pythoncom.CoUninitialize()

def default_features() -> dict:
    base = {"tempo": 0.0, "energy_rms": 0.0, "loudness_a": -80.0, "danceability": 0.0, "zcr": 0.0, "spectral_flux": 0.0, "spectral_centroid": 0.0, "spectral_rolloff": 0.0}
    base.update({k: 0.0 for k in BAND_KEYS})
    base.update({f"mfcc_{i+1}": 0.0 for i in range(20)})
    return base