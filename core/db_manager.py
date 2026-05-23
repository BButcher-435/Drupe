import sqlite3
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "presets.db")

JSON_PATH = os.path.join(BASE_DIR, "presets.json")
if not os.path.exists(JSON_PATH):
    JSON_PATH = os.path.join(BASE_DIR, "core", "presets.json")

FREQUENCIES = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000]

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Veritabanını ve Tabloyu Oluşturur. Boşsa JSON'dan fabrika ayarlarını çeker."""
    conn = get_connection()
    cursor = conn.cursor()
    columns = ", ".join([f"band_{freq} REAL" for freq in FREQUENCIES])
    
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            isim TEXT UNIQUE,
            {columns}
        )
    ''')
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM presets")
    if cursor.fetchone()[0] == 0:
        _load_defaults_from_json(conn)
        
    conn.close()

def _load_defaults_from_json(conn):
    """presets.json dosyasını okuyup veritabanına aktarır."""
    if not os.path.exists(JSON_PATH):
        print("Uyarı: presets.json bulunamadı, fabrika ayarları yüklenemedi.")
        return
        
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        presets_data = json.load(f)
        
    cursor = conn.cursor()
    for preset_name, bands in presets_data.items():
        band_values = []
        
        # JSON'daki verinin 15 elemanlı bir liste olduğunu kontrol et
        if isinstance(bands, list) and len(bands) == 15:
            band_values = [float(v) for v in bands]
        elif isinstance(bands, dict):
            for freq in FREQUENCIES:
                val = bands.get(str(freq), bands.get(freq, 0.0))
                band_values.append(float(val))
        else:
            band_values = [0.0] * 15 # Hata durumunda sıfırla
            
        placeholders = ", ".join(["?"] * 15)
        query = f"INSERT OR IGNORE INTO presets (isim, {', '.join([f'band_{f}' for f in FREQUENCIES])}) VALUES (?, {placeholders})"
        cursor.execute(query, [preset_name] + band_values)
        
    conn.commit()

def get_all_presets():
    """Tüm presetleri arayüzde kullanmak üzere sözlük (dict) formatında döndürür."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM presets")
    rows = cursor.fetchall()
    conn.close()
    
    result = {}
    for row in rows:
        isim = row[1]
        # row[2:] kısmı 15 frekansın değerleridir
        bands_dict = {freq: val for freq, val in zip(FREQUENCIES, row[2:])}
        result[isim] = bands_dict
        
    return result

def add_custom_preset(isim, bands_dict):
    """Kullanıcının arayüzden yaptığı manuel EQ ayarını veritabanına kaydeder."""
    conn = get_connection()
    cursor = conn.cursor()
    
    band_values = [bands_dict.get(freq, 0.0) for freq in FREQUENCIES]
    placeholders = ", ".join(["?"] * 15)
    query = f"REPLACE INTO presets (isim, {', '.join([f'band_{f}' for f in FREQUENCIES])}) VALUES (?, {placeholders})"
    
    cursor.execute(query, [isim] + band_values)
    conn.commit()
    conn.close()