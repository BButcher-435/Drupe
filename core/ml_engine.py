import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ── GTZAN 10 türünü bizim 7 EQ türüne eşle ──
GENRE_MAP = {
    "rock": "Rock",
    "metal": "Rock",
    "hiphop": "Hip-Hop",
    "pop": "Pop",
    "disco": "Pop",
    "reggae": "Pop",
    "country": "Folk",
    "blues": "Jazz",
    "jazz": "Jazz",
    "classical": "Classical",
    # Electronic GTZAN'da yok, disco zaten Pop'a gitti
}

EQ_PROFILES = {
    "Rock":       {"bass": 4,  "mid": -2, "treble": 3},
    "Electronic": {"bass": 6,  "mid": -3, "treble": 2},
    "Pop":        {"bass": 2,  "mid": 5,  "treble": 1},
    "Hip-Hop":    {"bass": 5,  "mid": 1,  "treble": 0},
    "Jazz":       {"bass": 0,  "mid": 4,  "treble": 3},
    "Classical":  {"bass": -1, "mid": 3,  "treble": 4},
    "Folk":       {"bass": 1,  "mid": 3,  "treble": 2},
}

GTZAN_PATH = "DataSets/genres_original"
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3  # her şarkıyı 3 saniyelik parçalara böl

_model = None
_label_encoder = None
_scaler = None

def get_model():
    global _model, _label_encoder, _scaler
    if _model is None:
        _model = joblib.load("core/model.pkl")
        _label_encoder = joblib.load("core/label_encoder.pkl")
        _scaler = joblib.load("core/scaler.pkl")
    return _model, _label_encoder, _scaler

def extract_features_from_audio(y, sr):
    """audio_processor.py ile BİREBİR aynı yöntemle özellik çıkarır."""
    N_FFT = 2048
    HOP_LENGTH = 512

    feats = []
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_power = np.abs(D) ** 2

    # MFCC — audio_processor ile AYNI: power_to_db üzerinden
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S_power, ref=np.max), sr=sr, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1))

    # ZCR
    feats.append(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)))

    # Spectral centroid
    feats.append(np.mean(librosa.feature.spectral_centroid(S=np.abs(D), sr=sr, n_fft=N_FFT)))

    # Spectral rolloff
    feats.append(np.mean(librosa.feature.spectral_rolloff(S=np.abs(D), sr=sr, n_fft=N_FFT, roll_percent=0.85)))

    # RMS energy
    feats.append(np.mean(librosa.feature.rms(y=y, hop_length=HOP_LENGTH)))

    # Spectral flux
    feats.append(np.mean(librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)))

    return feats

def model_egit():
    X = []
    y = []

    print("GTZAN ses dosyaları işleniyor...")

    for genre_folder in os.listdir(GTZAN_PATH):
        folder_path = os.path.join(GTZAN_PATH, genre_folder)
        if not os.path.isdir(folder_path):
            continue

        mapped_genre = GENRE_MAP.get(genre_folder.lower())
        if mapped_genre is None:
            continue

        print(f"  {genre_folder} → {mapped_genre}")

        for filename in os.listdir(folder_path):
            if not filename.endswith(".wav"):
                continue
            file_path = os.path.join(folder_path, filename)

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                segment_len = SEGMENT_SECONDS * sr

                # Şarkıyı 3 saniyelik parçalara böl
                num_segments = len(signal) // segment_len
                for s in range(num_segments):
                    start = s * segment_len
                    end = start + segment_len
                    segment = signal[start:end]

                    feats = extract_features_from_audio(segment, sr)
                    X.append(feats)
                    y.append(mapped_genre)
            except Exception as e:
                print(f"    Hata ({filename}): {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nToplam örnek: {len(X)}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Accuracy: {acc * 100:.1f}%")

    joblib.dump(scaler, "core/scaler.pkl")
    joblib.dump(model, "core/model.pkl")
    joblib.dump(le, "core/label_encoder.pkl")
    print("Model, scaler ve label encoder kaydedildi!")

def eq_hesapla(features: dict, override_genre: str = None) -> dict:
    """
    features: audio_processor'dan gelen sözlük (mfcc_1..20, zcr, spectral_centroid,
              spectral_rolloff, energy_rms, spectral_flux içermeli)
    override_genre: Apple/Spotify'dan tür geldiyse ML'i atla
    """
    model, le, scaler = get_model()

    # Modelin beklediği sırayla özellik vektörü oluştur
    feat_vector = []
    for i in range(1, 21):
        feat_vector.append(features.get(f"mfcc_{i}", 0.0))
    feat_vector.append(features.get("zcr", 0.0))
    feat_vector.append(features.get("spectral_centroid", 0.0))
    feat_vector.append(features.get("spectral_rolloff", 0.0))
    feat_vector.append(features.get("energy_rms", 0.0))
    feat_vector.append(features.get("spectral_flux", 0.0))

    X = np.array([feat_vector])
    X_scaled = scaler.transform(X)
    genre_encoded = model.predict(X_scaled)[0]
    ml_genre = le.inverse_transform([genre_encoded])[0]

    # Apple/Spotify türü varsa onu kullan, yoksa ML
    final_genre = override_genre if override_genre else ml_genre

    # EQ için basit yardımcı değerler (ses karakterinden)
    e = min(1.0, features.get("energy_rms", 0.05) * 5)
    centroid = features.get("spectral_centroid", 2000)
    zcr = features.get("zcr", 0.05)
    parlaklik = min(1.0, centroid / 4000)  # 0-1
    keskinlik = min(1.0, zcr * 8)           # 0-1

    base = EQ_PROFILES.get(final_genre, {"bass": 0, "mid": 0, "treble": 0})
    b = base["bass"]
    m = base["mid"]
    tr = base["treble"]

    bands = {
        25:    round(b * 0.5 + e * 2.0, 2),
        40:    round(b * 0.8 + e * 3.0, 2),
        63:    round(b * 1.0 + e * 3.5, 2),
        100:   round(b * 0.8 + e * 2.5, 2),
        160:   round(m * 0.5 + e * 1.0, 2),
        250:   round(m * 0.8, 2),
        400:   round(m * 1.0 - keskinlik * 1.0, 2),
        630:   round(m * 0.8, 2),
        1000:  round(m * 0.5 + keskinlik * 1.0, 2),
        1600:  round(tr * 0.5 + parlaklik * 1.5, 2),
        2500:  round(tr * 0.8 + parlaklik * 2.0, 2),
        4000:  round(tr * 1.0 + parlaklik * 2.0, 2),
        6300:  round(tr * 0.8 + parlaklik * 1.5, 2),
        10000: round(tr * 0.5 + parlaklik * 1.0, 2),
        16000: round(tr * 0.3 + parlaklik * 0.5, 2),
    }

    bands = {freq: max(-12, min(12, val)) for freq, val in bands.items()}

    return {
        "ml_prediction": ml_genre,
        "genre": final_genre,
        "bands": bands
    }

if __name__ == "__main__":
    model_egit()