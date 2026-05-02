import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import pandas as pd
from core.ml_engine import eq_hesapla

APO_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\config.txt"

def update_apo_config(bands_dict):
    eq_string = "GraphicEQ: "
    eq_parts = [f"{freq} {val}" for freq, val in bands_dict.items()]
    eq_string += "; ".join(eq_parts) + "\n"
    try:
        with open(APO_CONFIG_PATH, "w") as f:
            f.write(eq_string)
    except Exception as e:
        pass

def render():
    st.title("🎛️ Real Time EQ")
    st.markdown("Şu anda Spotify'da çalan şarkıyı dinliyor ve EQ ayarlarını anlık güncelliyor.")

    # Session state başlat
    if "eq_active" not in st.session_state:
        st.session_state.eq_active = False
    if "last_track_id" not in st.session_state:
        st.session_state.last_track_id = None

    # Butonlar
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Başlat", use_container_width=True):
            st.session_state.eq_active = True
            st.rerun()
    with col2:
        if st.button("⏹️ Durdur", use_container_width=True):
            st.session_state.eq_active = False
            st.rerun()

    st.divider()

    if st.session_state.eq_active:
        st.success("🟢 Dinamik EQ Aktif. Spotify dinleniyor...")

        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-read-currently-playing"))

        try:
            current_track = sp.current_user_playing_track()

            if current_track is not None and current_track.get('is_playing'):
                track_name = current_track['item']['name']
                artist_name = current_track['item']['artists'][0]['name']
                track_id = current_track['item']['id']
                img_url = current_track['item']['album']['images'][0]['url']

                audio_features = sp.audio_features(track_id)[0]

                if audio_features:
                    features_for_ml = {
                        "energy": audio_features["energy"],
                        "acousticness": audio_features["acousticness"],
                        "tempo": audio_features["tempo"],
                        "valence": audio_features["valence"],
                        "danceability": audio_features["danceability"],
                        "instrumentalness": audio_features["instrumentalness"],
                        "loudness": audio_features["loudness"],
                        "speechiness": audio_features["speechiness"]
                    }

                    # Yeni şarkıysa EQ güncelle
                    if track_id != st.session_state.last_track_id:
                        result = eq_hesapla(features_for_ml)
                        st.session_state.last_track_id = track_id
                        st.session_state.last_result = result
                        update_apo_config(result["bands"])
                    else:
                        result = st.session_state.last_result

                    # Şarkı bilgileri
                    img_col, info_col = st.columns([1, 2])
                    with img_col:
                        st.image(img_url, use_column_width=True)
                    with info_col:
                        st.subheader(track_name)
                        st.write(f"**Sanatçı:** {artist_name}")
                        st.write(f"**Tahmin Edilen Tür:** {result['genre']}")
                        st.write(f"**Tempo:** {audio_features['tempo']:.0f} BPM")
                        st.write(f"**Energy:** {audio_features['energy']:.2f}")

                    st.divider()

                    # EQ Grafiği
                    st.subheader("📊 EQ Ayarları")
                    bands = result["bands"]
                    df = pd.DataFrame({
                        "Frekans (Hz)": [str(k) for k in bands.keys()],
                        "dB": list(bands.values())
                    })
                    st.bar_chart(df.set_index("Frekans (Hz)"))

            else:
                st.info("Şu an Spotify'da müzik çalmıyor.")

        except Exception as e:
            st.error(f"Bağlantı veya model hatası: {e}")

        time.sleep(5)
        st.rerun()

    else:
        st.warning("🔴 EQ Kapalı. Başlatmak için yukarıdaki butona tıklayın.")