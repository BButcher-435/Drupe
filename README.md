# 🎧 EQUAI - Akıllı Spotify Equalizer

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Spotify](https://img.shields.io/badge/Spotify-1ED760?style=for-the-badge&logo=spotify&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

EQUAI, dinlediğiniz Spotify şarkılarını gerçek zamanlı olarak analiz eden ve Windows sisteminizdeki ses çıkışını makine öğrenmesi algoritmalarıyla otomatik olarak optimize eden yenilikçi bir Python projesidir.

---

##  Özellikler

* **Gerçek Zamanlı EQ (Real Time EQ):** Spotify'da çalan şarkıyı anlık olarak tespit eder, şarkının türünü makine öğrenmesi (Random Forest) ile tahmin eder ve Equalizer APO üzerinden sistem sesini dinamik olarak ayarlar.
* **Akıllı Playlist (Smart Playlist) (Yapım Aşamasında):** Şarkıların enerjisi ve BPM'ine göre yumuşak geçişler sağlayacak şekilde Spotify çalma listelerini vektörel uzaklık hesaplamaları (K-Means/TSP) kullanarak yeniden sıralar.
* **Modüler Mimari:** Geliştirilebilir, katmanlı proje yapısı.

---

##  Proje Mimarisi

Sistem 4 temel katmandan oluşmaktadır:
1.  **Kullanıcı Arayüzü:** Streamlit ile tasarlanmış hızlı ve interaktif web UI.
2.  **API & Yetkilendirme:** Spotipy kütüphanesi üzerinden güvenli OAuth2 entegrasyonu.
3.  **Veri İşleme & ML Motoru:** Şarkı verilerini işleyen, tür tahmini yapan ve ISO standartlarında 15 bantlık dB değerleri üreten Scikit-Learn pipeline'ı.
4.  **Yerel Donanım & Dışa Aktarım:** Hesaplanmış ayarları Windows `Equalizer APO` dizinine anında uygulayan modül.

---

##  Kurulum ve Çalıştırma

### Ön Koşullar
* Python 3.8 veya üzeri
* [Equalizer APO](https://sourceforge.net/projects/equalizerapo/) (Windows için ses motoru)
* Spotify Geliştirici Hesabı (API Anahtarları için)

### Kurulum Adımları

**1. Projeyi Klonlayın:**
```bash
git clone [https://github.com/BButcher-435/Drupe.git](https://github.com/BButcher-435/Drupe.git)
cd Drupe
```
**2. Sanal Ortam Oluşturun ve Bağımlılıkları Yükleyin:

```Bash
python -m venv venv
# Windows için:
.\venv\Scripts\activate
# Mac/Linux için:
source venv/bin/activate

pip install -r requirements.txt
```
3. Çevre Değişkenlerini Ayarlayın:
Proje ana dizininde bir .env dosyası oluşturun ve Spotify Developer Dashboard'dan aldığınız bilgileri girin:
```
SPOTIPY_CLIENT_ID='senin_client_id_kodun'
SPOTIPY_CLIENT_SECRET='senin_client_secret_kodun'
SPOTIPY_REDIRECT_URI='http://localhost:8501'
```
4. Uygulamayı Başlatın:

```Bash
streamlit run app.py
