import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import threading
import queue
import pandas as pd

from core.audio_processor import get_audio_state, audio_worker, default_features, BANDS, CHUNK_SECONDS
from core.eq_controller import update_apo_config
from core.ml_engine import eq_hesapla
from core import librosa_engine

@st.cache_resource
def get_cached_audio_state() -> dict:
    return get_audio_state()

def format_time(ms):
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    return f"{minutes}:{seconds:02d}"

def render():
    st.title("🎛️ Real Time Audio Analyzer & Player")
    st.markdown("Sistem sesini modüler olarak analiz eder, Spotify ile senkronize çalışır ve ML tabanlı dinamik EQ uygular.")

    state = get_cached_audio_state()
    
    if "eq_active" not in st.session_state:
        st.session_state.eq_active = False
    if "features" not in st.session_state:
        st.session_state.features = default_features()
    if "last_eq_state" not in st.session_state:
        st.session_state.last_eq_state = {freq: 0.0 for freq in librosa_engine.FREQUENCIES}
        st.session_state.features = default_features()
    if "last_spotify_time" not in st.session_state:
        st.session_state.last_spotify_time = 0
        st.session_state.spotify_data = None
        st.session_state.base_progress_ms = 0
    if "spotify_audio_features" not in st.session_state:  # ← YENİ
        st.session_state.spotify_audio_features = None
    if "last_track_id" not in st.session_state:           # ← YENİ
        st.session_state.last_track_id = None

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

        try:
            if time_since_fetch >= 3.5:
                scope = "user-read-currently-playing user-read-playback-state"
                sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
                playback = sp.current_playback()
                
                st.session_state.spotify_data = playback
                if playback and playback.get('is_playing'):
                    st.session_state.base_progress_ms = playback['progress_ms']
                    
                    # ── YENİ: Şarkı değişince audio features çek ──
                    track_id = playback['item']['id']
                    if track_id != st.session_state.last_track_id:
                        audio_features = sp.audio_features(track_id)[0]
                        if audio_features:
                            st.session_state.spotify_audio_features = audio_features
                            st.session_state.last_track_id = track_id
                    # ────────────────────────────────────────────────
                    
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

        try:
            st.session_state.features = state["result_queue"].get_nowait()
        except queue.Empty:
            pass 
        f = st.session_state.features

        # ── YENİ: Hibrit features (Spotify gerçek + Librosa anlık) ──
        spotify_af = st.session_state.spotify_audio_features
        if spotify_af:
            features_for_ml = {
                # Spotify'dan gerçek değerler
                "valence":          spotify_af["valence"],
                "acousticness":     spotify_af["acousticness"],
                "speechiness":      spotify_af["speechiness"],
                "instrumentalness": spotify_af["instrumentalness"],
                # Librosa'dan anlık değerler
                "energy":       min(f['energy_rms'] * 5, 1.0),
                "tempo":        f['tempo'] if f['tempo'] > 0 else spotify_af["tempo"],
                "danceability": f['danceability'] if f['danceability'] > 0 else spotify_af["danceability"],
                "loudness":     f['loudness_a'],
            }
        else:
            # Spotify henüz yüklenmediyse eski yöntemi kullan
            features_for_ml = {
                "energy":           min(f['energy_rms'] * 5, 1.0),
                "acousticness":     max(1.0 - (f['energy_rms'] * 5), 0.0),
                "tempo":            f['tempo'],
                "valence":          0.5,
                "danceability":     f['danceability'],
                "instrumentalness": 0.8 if f['zcr'] < 0.05 else 0.1,
                "loudness":         f['loudness_a'],
                "speechiness":      0.5 if f['zcr'] > 0.1 else 0.05
            }
        # ────────────────────────────────────────────────────────────
        
        try:
            result = eq_hesapla(features_for_ml)
            
            # ── YENİ: Spotify'dan genre varsa göster ──
            spotify_genre_info = ""
            if spotify_af:
                spotify_genre_info = " (Spotify + Librosa Hibrit)"
            st.success(f"🎸 ML Tahmini{spotify_genre_info}: **{result['genre']}**")
            # ────────────────────────────────────────────

            base_eq_bands = result["bands"]
            anlik_ses = {
                "rms":      f.get("energy_rms", 0.0),
                "zcr":      f.get("zcr", 0.0),
                "centroid": f.get("spectral_centroid", 0.0),
                "flux":     f.get("spectral_flux", 0.0)
            }
            
            hedef_eq = librosa_engine.apply_librosa_tweaks(base_eq_bands, anlik_ses)
            smoothed_eq = librosa_engine.apply_smoothing(st.session_state.last_eq_state, hedef_eq)
            st.session_state.last_eq_state = smoothed_eq.copy()
            
            preamp_val = librosa_engine.calculate_preamp(smoothed_eq)
            update_apo_config(smoothed_eq, preamp_val)

        except Exception as e:
            st.error(f"ML Motoru Hatası: {e}")

        with st.expander("📊 Gelişmiş Sistem Sesi Analizi (Canlı Veri)", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tempo", f"{f['tempo']:.1f} BPM")
            m2.metric("RMS Energy", f"{f['energy_rms']:.5f}")
            m3.metric("A-Weighted Loudness", f"{f['loudness_a']:.2f} dB")
            m4.metric("Danceability", f"{f['danceability']:.3f}")

            # ── YENİ: Spotify audio features göster ──
            if spotify_af:
                st.subheader("🎵 Spotify Audio Features (ML Girdileri)")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Valence", f"{spotify_af['valence']:.2f}")
                s2.metric("Acousticness", f"{spotify_af['acousticness']:.2f}")
                s3.metric("Speechiness", f"{spotify_af['speechiness']:.2f}")
                s4.metric("Instrumentalness", f"{spotify_af['instrumentalness']:.2f}")
            # ──────────────────────────────────────────

            st.subheader("Bant Güç Dağılımı")
            band_df = pd.DataFrame({
                "Band": [label for _, label, _, _ in BANDS],
                "Power": [f.get(key, 0.0) for key, _, _, _ in BANDS],
            }).set_index("Band")
            st.bar_chart(band_df, use_container_width=True, height=200)

            st.subheader("MFCC (Mel-Frequency Cepstral Coefficients)")
            mfcc_keys = [f"mfcc_{i+1}" for i in range(20)]
            mfcc_values = [f.get(key, 0.0) for key in mfcc_keys]
            mfcc_df = pd.DataFrame({
                "Coefficient": [f"C{i+1}" for i in range(20)],
                "Value": mfcc_values,
            }).set_index("Coefficient")
            st.bar_chart(mfcc_df, use_container_width=True, height=200)

        time.sleep(CHUNK_SECONDS)
        st.rerun()

    else:
        st.warning("🔴 Oynatıcı Kapalı. Başlatmak için yukarıdaki butona tıklayın.")