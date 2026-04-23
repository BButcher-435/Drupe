import streamlit as st
from dotenv import load_dotenv

# API anahtarlarını yükle
load_dotenv()

# Streamlit sayfa yapılandırması
st.set_page_config(page_title="EQUAI - Spotify EQ", page_icon="🎧", layout="wide")

def main():
    # Sadece yan menüyü (sidebar) çizer
    st.sidebar.title("🎧 EQUAI Menü")
    st.sidebar.markdown("---")
    
    # Kullanıcının mod seçimi yapması için menü
    app_mode = st.sidebar.radio(
        "Lütfen bir mod seçin:",
        ["Real Time EQ", "Smart Playlist"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Proje Modüllere Ayrılmış Mimariyle Çalışmaktadır.")

    # Sadece st.session_state veya seçim ile durum kontrolü yapıp views/ altındaki sayfaları çağırır
    if app_mode == "Real Time EQ":
        # views klasöründen realtime_eq dosyasını çağır
        from views import realtime_eq
        realtime_eq.render()  # realtime_eq.py içinde bir render() fonksiyonu olmalı
        
    elif app_mode == "Smart Playlist":
        # views klasöründen smart_playlist dosyasını çağır
        from views import smart_playlist
        smart_playlist.render() # smart_playlist.py içinde bir render() fonksiyonu olmalı

if __name__ == "__main__":
    main()