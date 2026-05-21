import os
APO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dynamic_eq.txt")
def update_apo_config(bands_dict: dict, preamp_val: float = 0.0):
    eq_string = f"Preamp: {preamp_val} dB\nGraphicEQ: " 
    eq_string += "; ".join([f"{freq} {val}" for freq, val in bands_dict.items()]) + "\n"
    try:
        with open(APO_CONFIG_PATH, "w") as f:
            f.write(eq_string)
    except Exception as e:
        print(f"Yazma Hatası: {e}")
def update_apo_config(bands_dict: dict):
    """Hesaplanmış frekans ve dB değerlerini Equalizer APO formatına çevirip uygular."""
    eq_string = "GraphicEQ: " + "; ".join([f"{freq} {val}" for freq, val in bands_dict.items()]) + "\n"
    try:
        with open(APO_CONFIG_PATH, "w") as f:
            f.write(eq_string)
    except Exception:
        pass
