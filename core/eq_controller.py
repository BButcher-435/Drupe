import os
import sys
import numpy as np
from .librosa_engine import apply_librosa_tweaks, apply_smoothing, apply_q_factor, calculate_preamp, FREQUENCIES

APO_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\config.txt"
LOCAL_BACKUP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dynamic_eq.txt")

current_bands = {freq: 0.0 for freq in FREQUENCIES}

def update_apo_config(bands_dict: dict, preamp_val: float = 0.0):
    """Equalizer APO config.txt'ye yazma ve yerel backup."""
    eq_lines = [f"Preamp: {preamp_val} dB"]
    eq_lines.append(f"GraphicEQ: {'; '.join([f'{freq} {val}' for freq, val in bands_dict.items()])}")

    eq_content = "\n".join(eq_lines) + "\n"

    try:
        with open(APO_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(eq_content)
        print(f"✓ APO Config yazıldı: {APO_CONFIG_PATH}")
    except FileNotFoundError:
        print(f"✗ APO config dosyası bulunamadı: {APO_CONFIG_PATH}")
    except PermissionError:
        print(f"✗ APO dosyasına yazma izni yok: {APO_CONFIG_PATH}")
    except Exception as e:
        print(f"✗ APO Yazma Hatası: {e}")

    try:
        with open(LOCAL_BACKUP_PATH, "w", encoding="utf-8") as f:
            f.write(eq_content)
    except Exception as e:
        print(f"✗ Yerel Backup Hatası: {e}")

def update_eq_from_audio_features(averaged_features: dict):
    """
    20 saniyelik ortalama audio feature'larından EQ bantlarını hesaplayıp,
    Equalizer APO config'e yazar.
    """
    global current_bands

    try:
        librosa_metrics = {
            "rms": averaged_features.get("energy_rms", 0.05),
            "zcr": averaged_features.get("zcr", 0.0),
            "centroid": averaged_features.get("spectral_centroid", 2000),
            "flux": averaged_features.get("spectral_flux", 0.0),
            "danceability": averaged_features.get("danceability", 0.0),
            "loudness": averaged_features.get("loudness_a", -80.0),
        }

        base_bands = {freq: 0.0 for freq in FREQUENCIES}
        tweaked_bands = apply_librosa_tweaks(base_bands, librosa_metrics)
        smoothed_bands = apply_smoothing(current_bands, tweaked_bands, alpha_attack=0.2, alpha_release=0.05)
        current_bands = smoothed_bands.copy()
        preamp = calculate_preamp(current_bands)

        print(f"📊 EQ Güncellendi - Preamp: {preamp} dB")
        update_apo_config(current_bands, preamp)

        return current_bands, preamp

    except Exception as e:
        print(f"✗ EQ Güncelleme Hatası: {e}")
        import traceback
        traceback.print_exc()
        return None, None