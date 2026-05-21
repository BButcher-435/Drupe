# 🎧 EquAI - Akıllı Ses Motoru ve Dinamik Equalizer (Real-Time DSP & ML)

![Python](https://img.shields.io/badge/python-3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

**EquAI**, bilgisayarınızda çalan sistem sesini (Spotify, YouTube, Oyunlar vb.) "Loopback" yöntemiyle anlık olarak dinleyen, Librosa ile sesin 20'den fazla dijital özelliğini (MFCC, RMS, ZCR, Centroid) çıkartan ve Makine Öğrenmesi (Random Forest) kullanarak Equalizer APO üzerinden **gerçek zamanlı (Real-Time) EQ optimizasyonu** yapan gelişmiş bir Python uygulamasıdır.

Sadece yapay zekaya bağlı kalmaz; içerdiği **SQLite destekli Manuel EQ motoru** sayesinde kendi ses profillerinizi oluşturmanıza, Apple Music/Spotify tarzı bir arayüzle frekanslara müdahale etmenize ve favori ayarlarınızı kaydetmenize olanak tanır.

---

## 🚀 Öne Çıkan Özellikler

* 🧠 **Yapay Zeka Destekli Otomatik EQ:** Sistem sesini her saniye analiz eder, çalan müziğin türünü tahmin eder ve seste patlamayı önleyen (Preamp) yumuşak geçişli (Smoothing) bir ISO standart EQ profili uygular.
* 🎛️ **Gelişmiş Manuel Kontrol (SQLite):** 15 bantlık dikey slider mimarisiyle sese manuel müdahale edin. Hazır profilleri (Rock, Pop, Jazz vb.) kullanın veya kendi ayarlarınızı veritabanına kaydedin.
* 🎧 **Evrensel Ses Yakalama:** `soundcard` kütüphanesi sayesinde sanal kablolara ihtiyaç duymadan Bluetooth kulaklıklar (örn: Marshall Major V) dahil tüm ses çıkışlarını kayıpsız yakalar.
* 🛡️ **Güvenli APO Entegrasyonu:** Windows Yönetici (Admin) izinlerine takılmamak için ayarları direkt kendi proje dizinindeki `dynamic_eq.txt` dosyasına yazar.
* 📊 **Canlı DSP Metrikleri:** Streamlit arayüzü üzerinden müziğin temposunu (BPM), A-Weighted Loudness, Danceability ve anlık spektral güç dağılımını canlı grafiklerle izleyin.

---

## 🏗️ Proje Mimarisi

EquAI, modülerliği ve sürdürülebilirliği sağlamak için katmanlı bir mimari kullanır:
* **`/core` (Çekirdek Motorlar):**
  * `audio_processor.py`: Projenin "Kulağı". C-Level DLL bağlantılarıyla sistem sesini dinler ve özellikleri çıkarır.
  * `ml_engine.py` & `librosa_engine.py`: Projenin "Beyni". Makine öğrenmesi tahminlerini yapar ve DSP matematiğini (Q-Factor, Smoothing) hesaplar.
  * `db_manager.py`: SQLite veritabanı yöneticisi.
  * `eq_controller.py`: Projenin "Eli". Hesaplanmış verileri APO konfigürasyonuna yazar.
* **`/views` (Arayüzler):** Streamlit modülleri (Real-Time EQ, Manuel EQ, Smart Playlist).

---

## 🛠️ Kurulum ve Çalıştırma

### ⚠️ Ön Koşullar (Zorunlu)
1. **Python 3.11** (Windows DLL uyumluluğu nedeniyle 3.12 veya 3.13 önerilmez).
2. [Equalizer APO](https://sourceforge.net/projects/equalizerapo/) Windows sisteminizde kurulu olmalıdır.

### Adım 1: Projeyi Klonlama ve Ortam Kurulumu
```bash
git clone [https://github.com/BButcher-435/Drupe.git](https://github.com/BButcher-435/Drupe.git)
cd Drupe
python -3.11 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
