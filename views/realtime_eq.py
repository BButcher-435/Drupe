import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import threading
import queue
import pandas as pd
import requests  # YENİ KURTARICIMIZ: İnternetten veri çekmek için

# ── MODÜLER MİMARİ İÇE AKTARIMLARI ──
from core.audio_processor import get_audio_state, audio_worker, default_features, BANDS, CHUNK_SECONDS
from core.eq_controller import update_apo_config
from core.ml_engine import eq_hesapla

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
    
    if any(keyword in genres_str for keyword in ["metal", "rock", "grunge", "punk", "anatolian"]): return "Rock"
    elif any(keyword in genres_str for keyword in ["hip hop", "rap", "trap", "drill"]): return "Hip-Hop"
    elif any(keyword in genres_str for keyword in ["pop", "dance", "r&b", "soul"]): return "Pop"
    elif any(keyword in genres_str for keyword in ["edm", "techno", "house", "electro", "dubstep"]): return "Electronic"
    elif any(keyword in genres_str for keyword in ["jazz", "blues"]): return "Jazz"
    elif any(keyword in genres_str for keyword in ["classical", "orchestra", "piano"]): return "Classical"
    elif any(keyword in genres_str for keyword in ["folk", "country", "acoustic", "indie"]): return "Folk"
    return None

def render():
    st.title("🎛️ Real Time Audio Analyzer & Player")
    st.markdown("Sistem sesini analiz eder ve **ML + Hibrit Veritabanı** ile dinamik EQ uygular.")

    state = get_cached_audio_state()
    
    if "current_track_id" not in st.session_state:
        st.session_state.current_track_id = None
        st.session_state.spotify_macro_genre = None
        st.session_state.raw_genres_debug = "Bekleniyor..."
        
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
                track_id = item['id']
                
                if st.session_state.current_track_id != track_id:
                    try:
                        if 'sp' not in locals():
                            scope = "user-read-currently-playing user-read-playback-state"
                            sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
                            
                        artist_id = item['artists'][0].get('id')
                        artist_name = item['artists'][0].get('name')
                        
                        raw_genres = []
                        debug_log = f"İsim: '{artist_name}'"
                        
                        # 1. ADIM: Spotify'a sor
                        if artist_id:
                            artist_info = sp.artist(artist_id)
                            raw_genres = artist_info.get('genres', [])
                            debug_log += f" | Spotify: {raw_genres}"
                        
                        # 2. ADIM (ÇÖZÜM): Spotify vermezse Apple/iTunes veritabanından SÖKE SÖKE AL!
                        if not raw_genres and artist_name:
                            try:
                                clean_name = artist_name.split("-")[0].split("(")[0].strip()
                                itunes_url = f"https://itunes.apple.com/search?term={clean_name}&entity=musicArtist&limit=1"
                                response = requests.get(itunes_url, timeout=3).json()
                                
                                if response['resultCount'] > 0:
                                    apple_genre = response['results'][0].get('primaryGenreName', '')
                                    raw_genres = [apple_genre.lower()]
                                    debug_log += f" | Apple/iTunes Kurtardı: {raw_genres}"
                                else:
                                    debug_log += " | Apple da bulamadı."
                            except Exception as e:
                                debug_log += f" | Apple Bağlantı Hatası."
                        
                        st.session_state.raw_genres_debug = debug_log
                        st.session_state.spotify_macro_genre = classify_spotify_genres(raw_genres)
                        
                        st.session_state.current_track_id = track_id
                    except Exception as inner_e:
                        st.session_state.raw_genres_debug = f"API Hatası: {inner_e}"
                
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

        features_for_ml = {
            "energy": min(f['energy_rms'] * 5, 1.0),
            "acousticness": max(1.0 - (f['energy_rms'] * 5), 0.0), 
            "tempo": f['tempo'],
            "valence": 0.5, 
            "danceability": f['danceability'],
            "instrumentalness": 0.8 if f['zcr'] < 0.05 else 0.1, 
            "loudness": f['loudness_a'],
            "speechiness": 0.5 if f['zcr'] > 0.1 else 0.05 
        }
        
        try:
            result = eq_hesapla(features_for_ml, st.session_state.spotify_macro_genre)
            
            st.success(f"🤖 ML: **{result['ml_prediction']}** | 🎧 Veritabanı: **{st.session_state.spotify_macro_genre or 'Belirsiz'}** | 🎛️ Uygulanan: **{result['genre']}**")
            st.caption(f"🔍 *Sistem Log:* `{st.session_state.raw_genres_debug}`")
            
            update_apo_config(result["bands"])
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