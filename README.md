# 🎧 EQUAI - Akıllı Equalizer (Real-Time Audio DSP & ML)

![Python](https://img.shields.io/badge/python-3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

EQUAI, bilgisayarınızda çalan **herhangi bir sesi (Spotify, YouTube, Oyunlar)** anlık olarak (Loopback ile) dinleyen, özellikleri (RMS, ZCR, MFCC vb.) Librosa ile çıkartan ve Makine Öğrenmesi (Random Forest) kullanarak Equalizer APO üzerinden **gerçek zamanlı (Real-Time) EQ optimizasyonu** yapan bir Python sistemidir.

---

## 🚀 Öne Çıkan Özellikler

* **Evrensel Ses Yakalama:** `soundcard` kütüphanesi sayesinde sanal kablolara (Stereo Mix) ihtiyaç duymadan Bluetooth kulaklıklar (örn: Major V) dahil tüm ses çıkışlarını doğrudan yakalar.
* **Canlı ML Tahmini:** Sesi her saniye analiz ederek türünü (Rock, Pop, Jazz vb.) tahmin eder ve 15 bantlık ISO standardı EQ profili çıkartır.
* **Dinamik DSP Motoru:** Sesteki anlık değişimlere (Kick vuruşları, tiz artışları) tepki vererek sesi yumuşak geçişlerle (Attack/Release) ve Preamp korumasıyla şekillendirir.
* **Güvenli APO Yazımı:** Windows Admin izinlerine takılmamak için ayarları kendi dizinindeki `dynamic_eq.txt` dosyasına yazar.

---

## 🛠️ Kurulum ve Çalıştırma

### Ön Koşullar (Çok Önemli!)
1. **Python 3.11** (Güvenlik ve DLL uyumluluğu nedeniyle Python 3.13 DESTEKLENMEZ).
2. [Equalizer APO](https://sourceforge.net/projects/equalizerapo/) Windows'ta kurulu olmalıdır.

### Adım 1: Projeyi Klonlama ve Ortam Kurulumu
```bash
git clone [https://github.com/BButcher-435/Drupe.git](https://github.com/BButcher-435/Drupe.git)
cd Drupe
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
