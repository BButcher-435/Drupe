import streamlit as st
import os
from dotenv import load_dotenv
from core import db_manager

load_dotenv()
st.set_page_config(page_title="EQUAI", page_icon="🎧", layout="wide")

def main():
    st.sidebar.title("🎧 EQUAI Menü")
    st.sidebar.markdown("---")
    db_manager.init_db()
    app_mode = st.sidebar.radio(
        "Lütfen bir mod seçin:",
        ["Real Time EQ", "Manuel EQ", "Smart Playlist"] # <-- Manuel EQ eklendi
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Proje Modüllere Ayrılmış Mimariyle Çalışmaktadır.")
    
    if app_mode == "Real Time EQ":
        from views import realtime_eq
        realtime_eq.render()
        
    elif app_mode == "Manuel EQ":
        from views import manuel_eq
        manuel_eq.render()
        
    elif app_mode == "Smart Playlist":
        from views import smart_playlist
        smart_playlist.render()

if __name__ == "__main__":
    main()
print("👉 EKRAN ÇIKTISI (CLIENT ID):", os.getenv("SPOTIPY_CLIENT_ID"))