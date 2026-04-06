import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

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
        ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    tahmin = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, tahmin) * 100:.1f}%")

    pickle.dump(pipeline, open("core/model.pkl", "wb"))
    print("Model kaydedildi!")

def eq_hesapla(features: dict) -> dict:
    model = pickle.load(open("core/model.pkl", "rb"))

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

    genre = model.predict(X)[0]

    # Kısayollar
    e  = features["energy"]
    ac = features["acousticness"]
    t  = (features["tempo"] - 60) / 140
    v  = features["valence"]
    d  = features["danceability"]
    i  = features["instrumentalness"]
    s  = features["speechiness"]

    # 15 band EQ (ISO standart frekanslar)
    bands = {
        25:    round(e * 2.0 + d * 1.5 - ac * 1.0, 2),            # sub bass
        40:    round(e * 3.5 + d * 2.5 - ac * 1.5, 2),            # bass alt
        63:    round(e * 4.0 + d * 3.0 - ac * 2.0, 2),            # bass
        100:   round(e * 3.0 + d * 2.0 - ac * 1.0, 2),            # upper bass
        160:   round(e * 1.5 + ac * 1.0 + i * 1.0, 2),            # low mid
        250:   round(ac * 2.5 + i * 2.0 - e * 1.0, 2),            # mid alt
        400:   round(ac * 1.5 + i * 1.5 - s * 1.5, 2),            # mid
        630:   round(i * 1.5 - s * 1.5 + v * 1.0, 2),             # upper mid
        1000:  round(s * 1.5 + v * 1.5 - i * 1.0, 2),             # presence
        1600:  round(v * 2.0 + s * 1.5 - ac * 1.0, 2),            # upper presence
        2500:  round(v * 2.5 + t * 1.5 - ac * 1.5, 2),            # treble
        4000:  round(e * 1.5 + v * 2.0 + t * 2.0, 2),             # high treble
        6300:  round(e * 1.0 + v * 1.5 + t * 2.5 - s * 1.0, 2),   # air alt
        10000: round(v * 1.5 + t * 2.5 - ac * 1.0 - s * 1.5, 2),  # air
        16000: round(v * 2.0 + t * 1.5 - ac * 2.0, 2),            # brilliance
    }

    bands = {freq: max(-12, min(12, val)) for freq, val in bands.items()}

    return {
        "genre": genre,
        "bands": bands
    }

if __name__ == "__main__":
    model_egit()

