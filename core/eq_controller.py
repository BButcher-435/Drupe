import os

# İleride bunu config/settings.py içinden çekecek şekilde de ayarlayabilirsiniz
APO_CONFIG_PATH = r"C:\Program Files\EqualizerAPO\config\config.txt"

def update_apo_config(bands_dict: dict):
    """Hesaplanmış frekans ve dB değerlerini Equalizer APO formatına çevirip uygular."""
    eq_string = "GraphicEQ: " + "; ".join([f"{freq} {val}" for freq, val in bands_dict.items()]) + "\n"
    try:
        with open(APO_CONFIG_PATH, "w") as f:
            f.write(eq_string)
    except Exception:
        pass # İzin veya bulunamama hatalarını arayüzü çökertmemek için yoksayıyoruz