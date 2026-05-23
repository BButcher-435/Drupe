import streamlit as st
import time
from core import db_manager
from core.eq_controller import update_apo_config
from core.librosa_engine import calculate_preamp, FREQUENCIES

def render():
    st.title("🎛️ Manuel EQ Ayarları")
    st.markdown("Kendi kulak zevkinize göre frekansları ayarlayın veya hazır profillerden birini seçin.")

    # 1. Veritabanından presetleri çek
    all_presets = db_manager.get_all_presets()
    preset_names = list(all_presets.keys())

    # --- KESİN ÇÖZÜM: CALLBACK FONKSİYONU ---
    # Bu fonksiyon sadece açılır menüden yeni bir preset seçildiğinde çalışır.
    def apply_preset():
        selected = st.session_state.preset_selector
        if selected != "-- Seçiniz --":
            preset_values = all_presets[selected]
            # Tüm çubukların (slider) değerlerini hafızada zorla güncelliyoruz
            for freq in FREQUENCIES:
                st.session_state[f"manuel_band_{freq}"] = preset_values[freq]

    # Sayfa ilk açıldığında slider hafızaları boş olmasın diye 0.0 ile doldur
    for freq in FREQUENCIES:
        if f"manuel_band_{freq}" not in st.session_state:
            st.session_state[f"manuel_band_{freq}"] = 0.0

    # 3. Hazır Profil Seçici
    st.selectbox(
        "🎵 Hazır Bir Profil Seçin:", 
        ["-- Seçiniz --"] + preset_names,
        key="preset_selector",
        on_change=apply_preset # Kullanıcı seçim yaptığı an yukarıdaki fonksiyon tetiklenir!
    )

    st.divider()
    st.subheader("Frekans Ayarları (dB)")

    # 4. 15 Kolon oluştur (Yatay ve bitişik)
    cols = st.columns(len(FREQUENCIES), gap="small")
    
    new_bands = {}
    for i, (col, freq) in enumerate(zip(cols, FREQUENCIES)):
        with col:
            label = f"{freq}" if freq < 1000 else f"{freq//1000}k"
            
            # DİKKAT: value= parametresini tamamen sildik. 
            # Streamlit artık değeri sadece 'key' üzerinden (yani hafızadan) okuyacak.
            new_val = st.slider(
                label, 
                min_value=-12.0, 
                max_value=12.0, 
                step=0.5,
                key=f"manuel_band_{freq}"
            )
            new_bands[freq] = new_val

    # 5. APO'ya Yazma İşlemi (Gereksiz yazmaları önlemek için kontrol)
    if "last_written_bands" not in st.session_state:
        st.session_state.last_written_bands = {}

    if new_bands != st.session_state.last_written_bands:
        st.session_state.last_written_bands = new_bands.copy()
        preamp_val = calculate_preamp(new_bands)
        update_apo_config(new_bands, preamp_val)

    # 6. Yeni Preset Kaydetme Bölümü
    st.divider()
    st.subheader("💾 Bu Ayarı Yeni Profil Olarak Kaydet")
    
    col_name, col_btn = st.columns([3, 1])
    with col_name:
        new_preset_name = st.text_input("Profil için bir isim girin:", placeholder="Örn: Benim Ayarım 1")
    with col_btn:
        st.write("") 
        st.write("")
        if st.button("Profili Kaydet", use_container_width=True):
            if new_preset_name.strip() == "":
                st.error("Lütfen bir isim girin!")
            else:
                db_manager.add_custom_preset(new_preset_name.strip(), new_bands)
                st.success(f"'{new_preset_name}' profili kaydedildi!")
                st.rerun()