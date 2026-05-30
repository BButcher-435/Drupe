import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import threading
import queue
import pandas as pd
import requests

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

def classify_spotify_genres(genres_list):
    if not genres_list:
        return None
    genres_str = " ".join(genres_list).lower()
    if any(k in genres_str for k in ["metal", "rock", "grunge", "punk", "anatolian"]): return "Rock"
    elif any(k in genres_str for k in ["hip hop", "rap", "trap", "drill"]): return "Hip-Hop"
    elif any(k in genres_str for k in ["pop", "dance", "r&b", "soul"]): return "Pop"
    elif any(k in genres_str for k in ["edm", "techno", "house", "electro", "dubstep"]): return "Electronic"
    elif any(k in genres_str for k in ["jazz", "blues"]): return "Jazz"
    elif any(k in genres_str for k in ["classical", "orchestra", "piano"]): return "Classical"
    elif any(k in genres_str for k in ["folk", "country", "acoustic", "indie"]): return "Folk"
    return None

def render():
    st.title("🎛️ Real Time Audio Analyzer & Player")
    st.markdown("Sistem sesini analiz eder, Spotify ile senkronize çalışır ve ML tabanlı dinamik EQ uygular.")

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
    if "spotify_audio_features" not in st.session_state:
        st.session_state.spotify_audio_features = None
    if "last_track_id" not in st.session_state:
        st.session_state.last_track_id = None
    if "spotify_macro_genre" not in st.session_state:
        st.session_state.spotify_macro_genre = None
    if "raw_genres_debug" not in st.session_state:
        st.session_state.raw_genres_debug = "Bekleniyor..."
    if "locked_genre" not in st.session_state:
        st.session_state.locked_genre = None
    if "locked_ml_prediction" not in st.session_state:
        st.session_state.locked_ml_prediction = None
    # ── Şarkı karakteri için veri biriktirme ──
    if "feature_buffer" not in st.session_state:
        st.session_state.feature_buffer = []
    if "collect_start_time" not in st.session_state:
        st.session_state.collect_start_time = None

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
                sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, open_browser=True))
                playback = sp.current_playback()

                st.session_state.spotify_data = playback
                if playback and playback.get('is_playing'):
                    st.session_state.base_progress_ms = playback['progress_ms']

                    track_id = playback['item']['id']
                    if track_id != st.session_state.last_track_id:

                        # Spotify audio features (genelde boş gelir, sorun değil)
                        try:
                            audio_features = sp.audio_features(track_id)[0]
                            if audio_features:
                                st.session_state.spotify_audio_features = audio_features
                        except Exception:
                            st.session_state.spotify_audio_features = None

                        # Apple iTunes tür kurtarıcı
                        artist_id = playback['item']['artists'][0].get('id')
                        artist_name = playback['item']['artists'][0].get('name')
                        raw_genres = []
                        debug_log = f"İsim: '{artist_name}'"

                        if artist_id:
                            artist_info = sp.artist(artist_id)
                            raw_genres = artist_info.get('genres', [])
                            debug_log += f" | Spotify: {raw_genres}"

                        if not raw_genres and artist_name:
                            try:
                                clean_name = artist_name.split("-")[0].split("(")[0].strip()
                                itunes_url = f"https://itunes.apple.com/search?term={clean_name}&entity=musicArtist&limit=1"
                                response = requests.get(itunes_url, timeout=3).json()
                                if response['resultCount'] > 0:
                                    apple_genre = response['results'][0].get('primaryGenreName', '')
                                    raw_genres = [apple_genre.lower()]
                                    debug_log += f" | Apple: {raw_genres}"
                                else:
                                    debug_log += " | Apple da bulamadı."
                            except Exception:
                                debug_log += " | Apple Bağlantı Hatası."

                        st.session_state.raw_genres_debug = debug_log
                        st.session_state.spotify_macro_genre = classify_spotify_genres(raw_genres)

                        # Yeni şarkı → kilidi ve buffer'ı sıfırla
                        st.session_state.locked_genre = None
                        st.session_state.locked_ml_prediction = None
                        st.session_state.collect_start_time = None
                        st.session_state.feature_buffer = []

                        st.session_state.last_track_id = track_id

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
                st.info("Şu an Spotify'da müzik çalmıyor (Sistem sesi dinleniyor).")
        except Exception:
            st.warning("Spotify bilgileri çekilemiyor, ses analizi devam ediyor...")

        try:
            st.session_state.features = state["result_queue"].get_nowait()
        except queue.Empty:
            pass
        f = st.session_state.features

        try:
            # ── ŞARKININ KARAKTERİNE GÖRE TEK TAHMİN ──
            COLLECT_SECONDS = 12

            if st.session_state.locked_genre is None:
                if st.session_state.collect_start_time is None:
                    st.session_state.collect_start_time = time.time()
                    st.session_state.feature_buffer = []

                # Anlamlı ses varsa ham MFCC sözlüğünü biriktir
                if f['energy_rms'] > 0.005:
                    st.session_state.feature_buffer.append(f.copy())

                elapsed = time.time() - st.session_state.collect_start_time

                if elapsed >= COLLECT_SECONDS and len(st.session_state.feature_buffer) >= 5:
                    # Biriken MFCC'lerin ortalamasını al
                    avg_features = {}
                    keys = st.session_state.feature_buffer[0].keys()
                    for key in keys:
                        try:
                            vals = [b[key] for b in st.session_state.feature_buffer]
                            avg_features[key] = sum(vals) / len(vals)
                        except (TypeError, KeyError):
                            avg_features[key] = st.session_state.feature_buffer[0][key]

                    result = eq_hesapla(avg_features, st.session_state.spotify_macro_genre)
                    st.session_state.locked_genre = result["genre"]
                    st.session_state.locked_ml_prediction = result["ml_prediction"]
                    base_eq_bands = result["bands"]
                else:
                    result = eq_hesapla(f, st.session_state.spotify_macro_genre)
                    base_eq_bands = result["bands"]
            else:
                result = eq_hesapla(f, st.session_state.locked_genre)
                base_eq_bands = result["bands"]

            # Ekran
            if st.session_state.locked_genre is None:
                st.info("🎧 Şarkı analiz ediliyor...")
            else:
                if st.session_state.spotify_macro_genre:
                
                    st.success(
                        f"🎧 Tür (Apple/Spotify): **{st.session_state.locked_genre}** | "
                        f"🎛️ Uygulanan EQ: **{st.session_state.locked_genre}**"
                    )
                else:
                    
                    st.success(
                        f"🤖 Tür (ML Tahmini): **{st.session_state.locked_genre}** | "
                        f"🎛️ Uygulanan EQ: **{st.session_state.locked_genre}**"
                    )
            st.caption(f"🔍 *Log:* `{st.session_state.raw_genres_debug}`")
           
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