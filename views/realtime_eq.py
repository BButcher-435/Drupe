import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import pandas as pd
import threading
from core.ml_engine import eq_hesapla
from core import librosa_engine # SENİN MOTORUN

def render():
    st.title("🎛️ Real Time EQ")
    st.markdown("Şu anda Spotify'da çalan şarkıyı dinliyor, Librosa ile analiz ediyor ve EQ ayarlarını anlık güncelliyor.")

    # 1. MOTORU ARKA PLANDA BAŞLAT (Sadece bir kere çalışması için)
    if "ses_motoru_aktif" not in st.session_state:
        st.session_state.ses_motoru_aktif = True
        motor_thread = threading.Thread(target=librosa_engine.start_engine, daemon=True)
        motor_thread.start()
        print("Streamlit: Ses Motoru Arka Planda Ateşlendi!")

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

        # Spotify API Bağlantısı (Kimlik doğrulaması tarayıcıda açılır)
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

                    # YENİ ŞARKI GELDİYSE (SADECE O ZAMAN MOTORU GÜNCELLE)
                    if track_id != st.session_state.last_track_id:
                        result = eq_hesapla(features_for_ml)
                        st.session_state.last_track_id = track_id
                        st.session_state.last_result = result
                        
                        # APO'YA YAZMA İŞLEMİNİ SENİN MOTORUNA DEVREDİYORUZ
                        librosa_engine.update_base_preset(result["bands"])
                    else:
                        result = st.session_state.last_result

                    # Şarkı Bilgilerini Ekrana Bas
                    img_col, info_col = st.columns([1, 2])
                    with img_col:
                        st.image(img_url, use_column_width=True)
                    with info_col:
                        st.subheader(track_name)
                        st.write(f"**Sanatçı:** {artist_name}")
                        st.write(f"**ML Tür Tahmini:** {result['genre']}")
                        st.write(f"**Tempo:** {audio_features['tempo']:.0f} BPM")

                    st.divider()

                    # EQ Grafiği (Arayüzde Base Preset Gösterilir)
                    st.subheader("📊 Taban EQ Profili (Base Preset)")
                    bands = result["bands"]
                    df = pd.DataFrame({
                        "Frekans (Hz)": [str(k) for k in bands.keys()],
                        "dB": list(bands.values())
                    })
                    st.bar_chart(df.set_index("Frekans (Hz)"))

            else:
                st.info("Şu an Spotify'da müzik çalmıyor.")

        except Exception as e:
            st.error(f"Bağlantı veya yetkilendirme hatası: {e}")

        # Her 5 saniyede bir Spotify'ı tekrar kontrol et
        time.sleep(5)
        st.rerun()

    else:
        st.warning("🔴 EQ Kapalı. Başlatmak için yukarıdaki butona tıklayın.")