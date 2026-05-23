import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---- EQ PROFİLLERİ ----
EQ_PROFILES = {
    "Rock":       {"bass": 4,  "mid": -2, "treble": 3},
    "Electronic": {"bass": 6,  "mid": -3, "treble": 2},
    "Pop":        {"bass": 2,  "mid": 5,  "treble": 1},
    "Hip-Hop":    {"bass": 5,  "mid": 1,  "treble": 0},
    "Jazz":       {"bass": 0,  "mid": 4,  "treble": 3},
    "Classical":  {"bass": -1, "mid": 3,  "treble": 4},
    "Folk":       {"bass": 1,  "mid": 3,  "treble": 2},
}

# ---- MODEL TEK SEFERİNDE YÜKLENİR (RAM DOSTU) ----
_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load("core/model.pkl")
    return _model

# ---- EĞİTİM ----
def model_egit():
    df = pd.read_csv("DataSets/songs.csv")

    genre_map = {
        "Rock": "Rock", "Electronic": "Electronic",
        "Pop": "Pop", "Hip-Hop": "Hip-Hop", "R&B": "Hip-Hop",
        "Jazz": "Jazz", "Blues": "Jazz",
        "Classical": "Classical", "Folk": "Folk", "Country": "Folk",
    }
    df["genre"] = df["genre"].map(genre_map)

    features = ["energy", "acousticness", "tempo", "valence",
                "danceability", "instrumentalness", "loudness", "speechiness"]
    df = df[features + ["genre"]].dropna()

    X = df[features]
    y = df["genre"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    tahmin = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, tahmin) * 100:.1f}%")

    joblib.dump(pipeline, "core/model.pkl")
    print("Model başarıyla kaydedildi (joblib)!")

# Fonksiyona "spotify_genre" adında opsiyonel bir parametre ekliyoruz
def eq_hesapla(features: dict, spotify_genre: str = None) -> dict:
    model = get_model()  # artık her seferinde yüklenmiyor

    X = [[
        features["energy"],
        features["acousticness"],
        features["tempo"],
        features["valence"],
        features["danceability"],
        features["instrumentalness"],
        features["loudness"],
        features["speechiness"]
    ]]

    # Modelin sadece sese bakarak yaptığı saf tahmin
    ml_genre = model.predict(X)[0]

    # KOPYA KAĞIDI MANTIĞI: Spotify'dan tür geldiyse onu, gelmediyse ML'in tahminini kullan
    final_genre = spotify_genre if spotify_genre else ml_genre

    e  = features["energy"]
    ac = features["acousticness"]
    t  = (features["tempo"] - 60) / 140
    v  = features["valence"]
    d  = features["danceability"]
    i  = features["instrumentalness"]
    s  = features["speechiness"]

    # Profil seçerken artık 'final_genre' kullanıyoruz
    base = EQ_PROFILES.get(final_genre, {"bass": 0, "mid": 0, "treble": 0})
    b = base["bass"]
    m = base["mid"]
    tr = base["treble"]

    bands = {
        25:    round(b * 0.5 + e * 2.0 + d * 1.5 - ac * 1.0, 2),
        40:    round(b * 0.8 + e * 3.5 + d * 2.5 - ac * 1.5, 2),
        63:    round(b * 1.0 + e * 4.0 + d * 3.0 - ac * 2.0, 2),
        100:   round(b * 0.8 + e * 3.0 + d * 2.0 - ac * 1.0, 2),
        160:   round(m * 0.5 + e * 1.5 + ac * 1.0 + i * 1.0, 2),
        250:   round(m * 0.8 + ac * 2.5 + i * 2.0 - e * 1.0, 2),
        400:   round(m * 1.0 + ac * 1.5 + i * 1.5 - s * 1.5, 2),
        630:   round(m * 0.8 + i * 1.5 - s * 1.5 + v * 1.0, 2),
        1000:  round(m * 0.5 + s * 1.5 + v * 1.5 - i * 1.0, 2),
        1600:  round(tr * 0.5 + v * 2.0 + s * 1.5 - ac * 1.0, 2),
        2500:  round(tr * 0.8 + v * 2.5 + t * 1.5 - ac * 1.5, 2),
        4000:  round(tr * 1.0 + e * 1.5 + v * 2.0 + t * 2.0, 2),
        6300:  round(tr * 0.8 + e * 1.0 + v * 1.5 + t * 2.5 - s * 1.0, 2),
        10000: round(tr * 0.5 + v * 1.5 + t * 2.5 - ac * 1.0 - s * 1.5, 2),
        16000: round(tr * 0.3 + v * 2.0 + t * 1.5 - ac * 2.0, 2),
    }

    bands = {freq: max(-12, min(12, val)) for freq, val in bands.items()}

    # Arayüzde iki tahmini de görebilmek için sonucu güncelledik
    return {
        "ml_prediction": ml_genre, 
        "genre": final_genre,
        "bands": bands
    }