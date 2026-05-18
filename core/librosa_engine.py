import librosa
import numpy as np
import sounddevice as sd  
from collections import deque # Hareketli ortalamalar (moving average) için performanslı liste
import threading # Sesi dinlerken sistemi/arayüzü dondurmamak için
import queue # Thread'ler arası güvenli veri aktarımı için
import time

# --- SES YAKALAMA SABİTLERİ ---
SAMPLE_RATE = 44100  # Standart ses kalitesi: Equalizer APO ile aynı olmalı = 44.1 kHz 
CHANNELS = 2         # Stereo ses için iki kanal
BUFFER_DURATION = 5  # Librosa'ya bir seferde kaç saniyelik ses gönderilecek? 
CHUNK_SIZE = int(SAMPLE_RATE * BUFFER_DURATION) # Tampon boyutu, büyük olması işlemci yükünü azaltır.

# - EQ BANTLARI -
FREQUENCIES = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000]

# --- GEÇMİŞ DURUM DEĞİŞKENLERİ (State) ---
# Smoothing (Yumuşak geçiş) için APO'ya yazılan bir önceki değerler burada tutalacak.
last_eq_state = {freq: 0.0 for freq in FREQUENCIES} 

# Librosa metriklerinin hareketli ortalamalarını tutacak kuyruklar.
history_rms = deque(maxlen=10) # En fazla 10 adet veriyi tutacak.
history_zcr = deque(maxlen=10)
history_centroid = deque(maxlen=10)

# Veri aktarım kuyruğu. Dinleyici thread'den hesaplayıcı thread'e ham ses verisi atmak için.
audio_queue = queue.Queue()

def calculate_preamp(target_bands): # preamp hesaplama fonksiyonu, taşma yaşanmaması için eklendi.
    max_gain = max(target_bands.values()) # en yüksek ses değerini bulur.
    preamp_val = min(0.0, -max_gain) 

    return round(preamp_val, 2)

def apply_q_factor(bands_dict, spread_factor=0.3): # Sivri EQ tepelerinin oluşmasını engeller. Seste Kalite faktörü demektir.
    smoothed_q_bands = bands_dict.copy()
    
    for i in range(len(FREQUENCIES)): # her frekans bandı için:
        current_freq = FREQUENCIES[i]
        
        # Eğer bu bantta bir değişiklik yapılmışsa (0 değilse)
        if abs(bands_dict[current_freq]) > 0.5: 
            # Sol komşu
            if i > 0:
                left_freq = FREQUENCIES[i-1]
                smoothed_q_bands[left_freq] += bands_dict[current_freq] * spread_factor
            
            # Sağ komşu
            if i < len(FREQUENCIES) - 1:
                right_freq = FREQUENCIES[i+1]
                smoothed_q_bands[right_freq] += bands_dict[current_freq] * spread_factor
                
    # Yayılım sonrası -12/+12 sınırlarını koru
    for freq in FREQUENCIES:
        smoothed_q_bands[freq] = max(-12.0, min(12.0, smoothed_q_bands[freq]))
        
    return smoothed_q_bands

# Smooth Geçiş 

def apply_smoothing(current_bands, target_bands, alpha_attack = 0.6, alpha_release = 0.1): 
    smoothed_bands = {}
    for freq in FREQUENCIES:
        eski = current_bands.get(freq, 0.0)
        hedef = target_bands.get(freq, 0.0)
        # Yönü belirle ve doğru alpha'yı seç
        if hedef > eski:
            # Sinyal artıyor (Attack devreye girer)
            yeni = (alpha_attack * hedef) + ((1.0 - alpha_attack) * eski)
        else:
            # Sinyal azalıyor (Release devreye girer)
            yeni = (alpha_release * hedef) + ((1.0 - alpha_release) * eski)
            
        smoothed_bands[freq] = round(yeni, 2)
        
    return smoothed_bands

# Librosa verileri

def apply_librosa_tweaks(base_bands, librosa_metrics):
    tweaked_bands = base_bands.copy() # iyileştirilmiş bantlar
    rms = librosa_metrics.get("rms", 0.05) # Anlık ses şiddeti 
    zcr = librosa_metrics.get("zcr", 0.0) # Ritmik yoğunluk / Davul (0 - 1 arası)
    centroid = librosa_metrics.get("centroid", 2000) # Sesin parlaklık merkezi (Hz)
    flux = librosa_metrics.get("flux", 0.0) # YENİ: Ani enerji değişimi
# --- KURAL 1: RMS (Ses Şiddeti) Duyarlılığı ---
    # Eğer şarkı çok sessizse (intro vb.), abartılı EQ yapmamak için etkiyi azalt.
    # RMS 0.1'den büyükse tam etki (1.0), düşükse azaltılmış etki.
    intensity_factor = min(1.0, max(0.1, rms * 10))
# --- KURAL 2: ZCR (Vurmalılar/Davullar) Desteği ---
    # ZCR yüksekse (şarkı hareketlendiyse) baslara ve alt-midlere vuruculuk (punch) ekle
# Kural 2.1: ZCR (Tiz / Trampet Vurmalılar)
    if zcr > 0.08: 
        tweaked_bands[2500] += (1.0 * intensity_factor) # Trampet/Snare gövdesi
        
    # Kural 2.2: FLUX (Bas/Kick Vuruşları ve Ani Patlamalar)
    if flux > 1.5: # Eşik değeri testlerle ayarlanır
        tweaked_bands[63] += (2.5 * intensity_factor) # Kick vuruculuğu (Punch)
        tweaked_bands[100] += (1.5 * intensity_factor)        
# --- KURAL 3: Spectral Centroid (Parlaklık/Yorgunluk) Kontrolü ---
    # Centroid 4000Hz üzerine çıkarsa, ses fazla tizleşmiş ve kulağı yoracak demektir. Tizleri biraz törpüle.
    if centroid > 4000:
        tweaked_bands[4000] -= (1.0 * intensity_factor)
        tweaked_bands[6300] -= (1.5 * intensity_factor)
        tweaked_bands[10000] -= (2.0 * intensity_factor)
# --- GÜVENLİK SINIRLANDIRMASI (Clipping Önlemi) ---
    # Tüm hesaplamalardan sonra, hiçbir bant -12'den küçük veya +12'den büyük olamaz. Bu yüzden gerekirse kesme işlemi yapılmalı.
    for freq in FREQUENCIES:
        tweaked_bands[freq] = max(-12.0, min(12.0, tweaked_bands[freq]))
        # Çıktıyı virgülden sonra 2 haneli olacak şekilde yuvarla
        tweaked_bands[freq] = round(tweaked_bands[freq], 2)

    tweaked_bands = apply_q_factor(tweaked_bands, spread_factor=0.35)
    return tweaked_bands

APO_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\config.txt"

# Burak'ın kodundan gelecek olan anlık taban EQ
aktif_base_bands = {freq: 0.0 for freq in FREQUENCIES}
global_hedef_eq = {freq: 0.0 for freq in FREQUENCIES}

def update_base_preset(yeni_base_bands):
    """Burak Enes'in kodu Spotify'dan yeni şarkı çekince bu fonksiyonu çağıracak."""
    global aktif_base_bands
    aktif_base_bands = yeni_base_bands.copy()
    print(">>> Spotify'dan Yeni Base Preset Geldi ve Motora Eklendi!")

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Ses Yakalama Hatası: {status}")
    audio_queue.put(indata.copy())

def audio_processing_thread():
    """Librosa'nın sesi analiz ettiği Makro Döngü (Her 5 saniyede bir yeni veri geldikçe)"""
    global global_hedef_eq, aktif_base_bands
    
    while True:
        audio_chunk = audio_queue.get() # Ses gelene kadar burada bekler
        mono_audio = np.mean(audio_chunk, axis=1) # Stereo'yu Mono'ya çevir
        mono_audio_light = librosa.resample(mono_audio, orig_sr=44100, target_sr=22050)
        rms_val = np.mean(librosa.feature.rms(y=mono_audio_light))
        try:
            # Sessizlik kontrolü (Ses yoksa boşuna işlemci yorma)
            rms_val = np.mean(librosa.feature.rms(y=mono_audio))
            #Python'un tam olarak ne duyduğunu görelim
            print(f"🎤 Dinlenen Anlık Ses Şiddeti (RMS): {rms_val:.5f}")
            if rms_val < 0.001:
                global_hedef_eq = aktif_base_bands.copy()
                continue
                
            zcr_val = np.mean(librosa.feature.zero_crossing_rate(y=mono_audio))
            centroid_val = np.mean(librosa.feature.spectral_centroid(y=mono_audio, sr=SAMPLE_RATE))
            flux_val = np.mean(librosa.onset.onset_strength(y=mono_audio, sr=SAMPLE_RATE))
            
            history_rms.append(rms_val)
            history_zcr.append(zcr_val)
            history_centroid.append(centroid_val)
            
            avg_metrics = {
                "rms": np.mean(history_rms),
                "zcr": np.mean(history_zcr),
                "centroid": np.mean(history_centroid),
                "flux": flux_val 
            }
            global_hedef_eq = apply_librosa_tweaks(aktif_base_bands, avg_metrics)
            print(f"🎵 Analiz: RMS={avg_metrics['rms']:.3f}, ZCR={avg_metrics['zcr']:.3f} | APO'ya Gönderilen Hedef: {global_hedef_eq[63]} Hz (Bas)")
            
        except Exception as e:
            print(f"Librosa Analiz Hatası: {e}")

def eq_writing_thread():
    """Saniyede 20 kere APO'ya yumuşak geçişle yazan Mikro Döngü"""
    global last_eq_state, global_hedef_eq
    
    while True:
        # Smoothing (Yumuşak Geçiş)
        suanki_eq = apply_smoothing(last_eq_state, global_hedef_eq, alpha_attack=0.2, alpha_release=0.05)
        last_eq_state = suanki_eq.copy()
        
        # Preamp (Kırpılma Önleyici)
        preamp_val = calculate_preamp(suanki_eq)
        
        # APO'ya Yazma İşlemi
        try:
            eq_string = f"Preamp: {preamp_val} dB\nGraphicEQ: "
            eq_parts = [f"{freq} {val}" for freq, val in suanki_eq.items()]
            eq_string += "; ".join(eq_parts) + "\n"
            
            with open(APO_CONFIG_PATH, "w") as f:
                f.write(eq_string)
        except Exception as e:
            pass 
        time.sleep(0.1)
def start_engine():
    print("EquAI Ses Motoru Başlatılıyor...")
    
    writer_thread = threading.Thread(target=eq_writing_thread, daemon=True)
    writer_thread.start()
    
    analyzer_thread = threading.Thread(target=audio_processing_thread, daemon=True)
    analyzer_thread.start()
    
    print("Ses Dinleniyor... (Durdurmak için terminalde Ctrl+C yapın)")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE, device=1):
            import time
            while True:
                time.sleep(1) # Ctrl+C'yi algılayabilmek için basit bir döngü
    except KeyboardInterrupt:
        print("\nMotor Kapatıldı. Görüşmek üzere!")
if __name__ == "__main__":
    print("\n" + "="*40)
    print("🚀 PYTHON MOTORU TETİKLENDİ!")
    print("="*40 + "\n")
    
    test_preset = {freq: 0.0 for freq in FREQUENCIES}
    update_base_preset(test_preset)
    start_engine()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=CHUNK_SIZE, device=1):
        threading.Event().wait()