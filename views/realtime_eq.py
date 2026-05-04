import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import numpy as np
import soundcard as sc
import librosa
import threading
import queue
import pythoncom
import pandas as pd
from core.ml_engine import eq_hesapla
from config.settings import APO_CONFIG_PATH

# ─────────────────────────────────────────────────────────────────
# SES ANALİZİ SABİTLERİ VE AYARLARI
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

@st.cache_resource
def get_state() -> dict:
    return {
        "result_queue": queue.Queue(maxsize=3),
        "stop_event":   threading.Event(),
        "ring_buffer":  np.zeros(BUFFER_FRAMES, dtype=np.float32),
        "ring_lock":    threading.Lock(),
        "ema":          {},
        "worker":       None,
    }

def _ema(store: dict, key: str, value: float) -> float:
    prev = store.get(key, value)
    v    = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * prev
    store[key] = v
    return float(v)

# ─────────────────────────────────────────────────────────────────
# ÖZELLİK ÇIKARIMI (FEATURE EXTRACTION) & THREAD
# ─────────────────────────────────────────────────────────────────
def extract_features(y: np.ndarray, ema_store: dict) -> dict:
    KEYS = ["tempo", "energy_rms", "loudness_a", "danceability", "zcr", "spectral_flux", "spectral_centroid", "spectral_rolloff", *BAND_KEYS]
    if float(np.max(np.abs(y))) < 1e-5:
        return {k: ema_store.get(k, 0.0) for k in KEYS}

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

    mfcc_mean = np.mean(librosa.feature.mfcc(S=librosa.power_to_db(S_power), sr=SAMPLE_RATE, n_mfcc=20), axis=1)
    mfcc_dict = {f"mfcc_{i+1}": _ema(ema_store, f"mfcc_{i+1}", float(val)) for i, val in enumerate(mfcc_mean)}

    total_pwr = float(S_power.mean()) + 1e-10
    band_out = {}
    for key, _, f_lo, f_hi in BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_out[key] = _ema(ema_store, key, float(S_power[mask].mean() / total_pwr) if mask.any() else 0.0)

    return {"tempo": tempo, "energy_rms": energy_rms, "loudness_a": loudness_a, "danceability": danceability, "zcr": zcr, "spectral_flux": flux, "spectral_centroid": spectral_centroid, "spectral_rolloff": spectral_rolloff, **band_out, **mfcc_dict}

def _worker(state: dict) -> None:
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
                if rq.full():
                    try: rq.get_nowait()
                    except queue.Empty: pass
                try: rq.put_nowait(feats)
                except queue.Full: pass
    finally:
        pythoncom.CoUninitialize()

def _default_features() -> dict:
    base = {"tempo": 0.0, "energy_rms": 0.0, "loudness_a": -80.0, "danceability": 0.0, "zcr": 0.0, "spectral_flux": 0.0, "spectral_centroid": 0.0, "spectral_rolloff": 0.0}
    base.update({k: 0.0 for k in BAND_KEYS})
    base.update({f"mfcc_{i+1}": 0.0 for i in range(20)})
    return base

# ─────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────
def format_time(ms):
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    return f"{minutes}:{seconds:02d}"

def update_apo_config(bands_dict):
    eq_string = "GraphicEQ: " + "; ".join([f"{freq} {val}" for freq, val in bands_dict.items()]) + "\n"
    try:
        with open(APO_CONFIG_PATH, "w") as f:
            f.write(eq_string)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────
# ANA ARAYÜZ (RENDER)
# ─────────────────────────────────────────────────────────────────
def render():
    st.title("🎛️ Real Time EQ & Audio Analyzer")
    st.markdown("Sistem sesini canlı analiz eder, Spotify oynatıcısıyla senkronize çalışır ve EQ'yu günceller.")

    state = get_state()
    
    if "eq_active" not in st.session_state:
        st.session_state.eq_active = False
    if "features" not in st.session_state:
        st.session_state.features = _default_features()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Başlat", use_container_width=True, disabled=st.session_state.eq_active):
            st.session_state.eq_active = True
            state["stop_event"].clear()
            state["ema"].clear()
            with state["ring_lock"]:
                state["ring_buffer"][:] = 0.0
            t = threading.Thread(target=_worker, args=(state,), daemon=True)
            state["worker"] = t
            t.start()
            st.rerun()
    with col2:
        if st.button("⏹️ Durdur", use_container_width=True, disabled=not st.session_state.eq_active):
            st.session_state.eq_active = False
            state["stop_event"].set()
            st.session_state.features = _default_features()
            st.rerun()

    st.divider()

    if st.session_state.eq_active:
        # 1. SPOTIFY GÖRSEL OYNATICI (Sadece Arayüz İçin)
        scope = "user-read-currently-playing user-read-playback-state"
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
        try:
            playback = sp.current_playback()
            if playback is not None and playback.get('is_playing'):
                item = playback['item']
                progress_ms = playback['progress_ms']
                duration_ms = item['duration_ms']
                
                img_col, info_col = st.columns([1, 2])
                with img_col:
                    st.image(item['album']['images'][0]['url'], use_column_width=True)
                with info_col:
                    st.subheader(item['name'])
                    with st.expander(f"🎤 **Sanatçı:** {item['artists'][0]['name']}"):
                        try:
                            genres = sp.artist(item['artists'][0]['id']).get('genres', [])
                            st.write(f"**Türler:** {', '.join(genres).title() if genres else 'Bulunamadı'}")
                        except: st.write("Bilgi çekilemedi.")
                    
                    st.progress(min(progress_ms / duration_ms, 1.0))
                    time_col1, time_col2 = st.columns(2)
                    time_col1.caption(f"▶ Çalınan: {format_time(progress_ms)}")
                    time_col2.caption(f"⏳ Kalan: {format_time(duration_ms - progress_ms)}")
            else:
                st.info("Şu an Spotify'da müzik çalmıyor (Ancak sistem sesi dinleniyor).")
        except Exception as e:
            st.warning(f"Spotify arayüzü yüklenemedi: {e}")

        # 2. GERÇEK ZAMANLI SES ANALİZİ (Yazdığınız Modül)
        try:
            st.session_state.features = state["result_queue"].get_nowait()
        except queue.Empty:
            pass 
        f = st.session_state.features

        # 3. MAKİNE ÖĞRENMESİ İÇİN KÖPRÜ (Mapping)
        # librosa değerlerini ML modelinizin beklediği değer aralıklarına yaklaştırıyoruz
        features_for_ml = {
            "energy": min(f['energy_rms'] * 5, 1.0), # Normalize denemesi
            "acousticness": max(1.0 - (f['energy_rms'] * 5), 0.0), 
            "tempo": f['tempo'],
            "valence": 0.5, # Librosa doğrudan vermediği için nötr bırakıyoruz
            "danceability": f['danceability'],
            "instrumentalness": 0.8 if f['zcr'] < 0.05 else 0.1, 
            "loudness": f['loudness_a'],
            "speechiness": 0.5 if f['zcr'] > 0.1 else 0.05 
        }
        
        result = eq_hesapla(features_for_ml)
        st.success(f"🎸 ML Tahmini (Sistem Sesinden): **{result['genre']}**")
        update_apo_config(result["bands"])

        # 4. GELİŞMİŞ SES ANALİZİ DASHBOARD'U
        with st.expander("📊 Gelişmiş Sistem Sesi Analizi (Canlı Veri)", expanded=True):
            st.caption(f"Window {BUFFER_SECONDS}s | Update {CHUNK_SECONDS}s | SR {SAMPLE_RATE}Hz")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tempo", f"{f['tempo']:.1f} BPM")
            m2.metric("RMS Energy", f"{f['energy_rms']:.5f}")
            m3.metric("A-Weighted Loudness", f"{f['loudness_a']:.2f} dB(A)")
            m4.metric("Danceability", f"{f['danceability']:.3f}")

            band_df = pd.DataFrame({
                "Band": [label for _, label, _, _ in BANDS],
                "Power Ratio": [f.get(key, 0.0) for key, _, _, _ in BANDS],
            }).set_index("Band")
            st.bar_chart(band_df, use_container_width=True, height=200)

        # Döngüyü canlı tut (0.4 saniyede bir arayüz yenilenir)
        time.sleep(CHUNK_SECONDS)
        st.rerun()

    else:
        st.warning("🔴 EQ ve Oynatıcı Kapalı. Başlatmak için yukarıdaki butona tıklayın.")