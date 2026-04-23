import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
from core.ml_engine import eq_hesapla

# Not: Şimdilik Equalizer APO yolunu buraya ekliyoruz.
# İlerleyen aşamalarda mimariye uyup bunu config/settings.py içine taşıyacağız.
APO_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\config.txt"

def update_apo_config(bands_dict):
    """31 bantlık EQ verisini config dosyasına yazar."""
    eq_string = "GraphicEQ: "
    eq_parts = [f"{freq} {val}" for freq, val in bands_dict.items()]
    eq_string += "; ".join(eq_parts) + "\n"
    try:
        with open(APO_CONFIG_PATH, "w") as f:
            f.write(eq_string)
    except Exception as e:
        pass # Arayüzü bozmamak için hataları şimdilik pass geçiyoruz

def render():
    st.title("🎛️ Real Time EQ")
    st.markdown("Şu anda Spotify'da çalan şarkıyı dinliyor ve EQ ayarlarını anlık güncelliyor.")

    # Buton durumunu aklında tutması için session_state kullanıyoruz
    if "eq_active" not in st.session_state:
        st.session_state.eq_active = False

    # Start ve Stop butonlarını yan yana koyalım
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Başlat", use_container_width=True):
            st.session_state.eq_active = True
            st.rerun() # Arayüzü anında güncelle
    with col2:
        if st.button("⏹️ Durdur", use_container_width=True):
            st.session_state.eq_active = False
            st.rerun()

    st.divider()

    # Eğer sistem aktifse Spotify'ı dinle
    if st.session_state.eq_active:
        st.success("🟢 Dinamik EQ Aktif. Spotify dinleniyor...")
        
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-read-currently-playing"))
        
        try:
            current_track = sp.current_user_playing_track()
            
            if current_track is not None and current_track.get('is_playing'):
                track_name = current_track['item']['name']
                artist_name = current_track['item']['artists'][0]['name']
                track_id = current_track['item']['id']
                
                # Kapak fotoğrafını al
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
                    
                    # Tür tahmini ve EQ hesabı
                    result = eq_hesapla(features_for_ml)
                    
                    # Kapak fotoğrafı ve şarkı bilgilerini yan yana göster
                    img_col, info_col = st.columns([1, 2])
                    with img_col:
                        st.image(img_url, use_column_width=True)
                    with info_col:
                        st.subheader(track_name)
                        st.write(f"**Sanatçı:** {artist_name}")
                        st.write(f"**Tahmin Edilen Tür:** {result['genre']}")
                        
                    # APO'yu güncelle
                    update_apo_config(result["bands"])
                    
            else:
                st.info("Şu an Spotify'da müzik çalmıyor.")
                
        except Exception as e:
            st.error(f"Bağlantı veya model hatası: {e}")
        
        # Sürekli dinleme efekti yaratmak için 5 saniye bekle ve sayfayı yenile
        time.sleep(5)
        st.rerun()
        
    else:
        st.warning("🔴 EQ Kapalı. Başlatmak için yukarıdaki butona tıklayın.")