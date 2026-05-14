import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import threading
import queue
import pandas as pd

# ── MODÜLER MİMARİ İÇE AKTARIMLARI ──
from core.audio_processor import get_audio_state, audio_worker, default_features, BANDS, CHUNK_SECONDS
from core.eq_controller import update_apo_config

@st.cache_resource
def get_cached_audio_state() -> dict:
    return get_audio_state()

def format_time(ms):
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    return f"{minutes}:{seconds:02d}"

def render():
    st.title("🎛️ Real Time Audio Analyzer & Player")
    st.markdown("Sistem sesini modüler olarak analiz eder ve Spotify ile senkronize çalışır. (ML Tahmini modülü geçici olarak devre dışıdır).")

    state = get_cached_audio_state()
    
    # Session State Başlangıçları
    if "eq_active" not in st.session_state:
        st.session_state.eq_active = False
    if "features" not in st.session_state:
        st.session_state.features = default_features()
    if "last_spotify_time" not in st.session_state:
        st.session_state.last_spotify_time = 0
        st.session_state.spotify_data = None
        st.session_state.base_progress_ms = 0

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Başlat", use_container_width=True, disabled=st.session_state.eq_active):
            st.session_state.eq_active = True
            state["stop_event"].clear()
            state["ema"].clear()
            with state["ring_lock"]:
                state["ring_buffer"][:] = 0.0
            t = threading.Thread(target=audio_worker, args=(state,), daemon=True)
            state["worker"] = t
            t.start()
            st.rerun()
    with col2:
        if st.button("⏹️ Durdur", use_container_width=True, disabled=not st.session_state.eq_active):
            st.session_state.eq_active = False
            state["stop_event"].set()
            st.session_state.features = default_features()
            st.rerun()

    st.divider()

    if st.session_state.eq_active:
        current_time = time.time()
        time_since_fetch = current_time - st.session_state.last_spotify_time

        # 1. SPOTIFY OYNATICI (Rate-Limit Korumalı)
        try:
            if time_since_fetch >= 3.5:
                scope = "user-read-currently-playing user-read-playback-state"
                sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
                playback = sp.current_playback()
                
                st.session_state.spotify_data = playback
                if playback and playback.get('is_playing'):
                    st.session_state.base_progress_ms = playback['progress_ms']
                st.session_state.last_spotify_time = current_time
            else:
                playback = st.session_state.spotify_data

            if playback is not None and playback.get('is_playing'):
                item = playback['item']
                duration_ms = item['duration_ms']
                
                calculated_progress = st.session_state.base_progress_ms + int(time_since_fetch * 1000)
                progress_ms = min(calculated_progress, duration_ms)
                
                img_col, info_col = st.columns([1, 2])
                with img_col:
                    st.image(item['album']['images'][0]['url'], use_column_width=True)
                with info_col:
                    st.subheader(item['name'])
                    st.caption(f"🎤 **Sanatçı:** {item['artists'][0]['name']}")
                    
                    st.progress(min(progress_ms / duration_ms, 1.0))
                    time_col1, time_col2 = st.columns(2)
                    time_col1.caption(f"▶ Çalınan: {format_time(progress_ms)}")
                    time_col2.caption(f"⏳ Kalan: {format_time(duration_ms - progress_ms)}")
            else:
                st.info("Şu an Spotify'da müzik çalmıyor (Ancak sistem sesi dinleniyor).")
        except Exception:
            st.warning("Spotify bilgileri şu an çekilemiyor, ancak ses analizi devam ediyor...")

        # 2. ARKA PLAN SES VERİSİNİ ÇEKME
        try:
            st.session_state.features = state["result_queue"].get_nowait()
        except queue.Empty:
            pass 
        f = st.session_state.features

        # ML TAHMİNİ GEÇİCİ OLARAK KAPALI - RAM TASARRUFU İÇİN SADECE FLAT EQ GÖNDERİLİYOR
        dummy_bands = {25: 0, 40: 0, 63: 0, 100: 0, 160: 0, 250: 0, 400: 0, 630: 0, 1000: 0, 1600: 0, 2500: 0, 4000: 0, 6300: 0, 10000: 0, 16000: 0}
        update_apo_config(dummy_bands)

        # 3. GELİŞMİŞ SES ANALİZİ DASHBOARD'U
        with st.expander("📊 Gelişmiş Sistem Sesi Analizi (Canlı Veri)", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tempo", f"{f['tempo']:.1f} BPM")
            m2.metric("RMS Energy", f"{f['energy_rms']:.5f}")
            m3.metric("A-Weighted Loudness", f"{f['loudness_a']:.2f} dB")
            m4.metric("Danceability", f"{f['danceability']:.3f}")

            # --- 15 BANT GÜÇ DAĞILIMI ---
            st.subheader("Bant Güç Dağılımı")
            band_df = pd.DataFrame({
                "Band": [label for _, label, _, _ in BANDS],
                "Power": [f.get(key, 0.0) for key, _, _, _ in BANDS],
            }).set_index("Band")
            st.bar_chart(band_df, use_container_width=True, height=200)

            # --- YENİ EKLENEN KISIM: MFCC GRAFİĞİ ---
            st.subheader("MFCC (Mel-Frequency Cepstral Coefficients)")
            mfcc_keys = [f"mfcc_{i+1}" for i in range(20)]
            mfcc_values = [f.get(key, 0.0) for key in mfcc_keys]
            
            mfcc_df = pd.DataFrame({
                "Coefficient": [f"C{i+1}" for i in range(20)],
                "Value": mfcc_values,
            }).set_index("Coefficient")
            
            # MFCC genelde negatif ve pozitif değerler içerdiği için bar chart çok uygundur
            st.bar_chart(mfcc_df, use_container_width=True, height=200)
            # ----------------------------------------

        time.sleep(CHUNK_SECONDS)
        st.rerun()

    else:
        st.warning("🔴 Oynatıcı Kapalı. Başlatmak için yukarıdaki butona tıklayın.")