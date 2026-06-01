#  EquAI - Smart Audio Engine & Dynamic Equalizer (Real-Time DSP & ML)

[![Python](https://img.shields.io/badge/python-3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![PyPI version](https://badge.fury.io/py/equai.svg)](https://pypi.org/project/equai/)
## 📖 Overview
**EquAI** is a cutting-edge, Python-based digital signal processing (DSP) application that actively monitors your system's audio output via Loopback. By extracting over 20 digital audio features (such as MFCC, RMS, ZCR, and Spectral Centroid) in real-time, EquAI leverages a Random Forest Machine Learning model to apply **dynamic EQ optimization** directly through Equalizer APO.

Whether you're listening to Spotify, watching YouTube, or gaming, EquAI acts as an intelligent layer that adapts your audio profile on the fly. 

---

##  Key Features

- 🧠 **AI-Powered Automated EQ:** Continuously analyzes system audio, predicts the music genre, and applies smooth, ISO-standard EQ transitions with Preamp limiters to prevent clipping.
- 🎛️ **Advanced Manual Control:** Take the wheel with a 15-band vertical slider architecture backed by SQLite. Load built-in profiles (Rock, Pop, Jazz) or save your custom sonic signatures.
- 🎧 **Universal Audio Capture:** Employs the `soundcard` library for lossless capture of all audio outputs—including Bluetooth headsets—eliminating the need for virtual audio cables.
- 🛡️ **Seamless APO Integration:** Directly writes configurations to `dynamic_eq.txt` within the project directory, cleanly bypassing strict Windows Administrator restrictions.
- 📊 **Live DSP Telemetry:** Monitor dynamic audio data via a sleek Streamlit interface. Track live BPM, A-Weighted Loudness, Danceability, and spectral power distribution with interactive charts.

---

##  Architecture

EquAI is built with a highly modular architecture to ensure low-latency performance and easy scalability:

- **`/core` (The Engines):**
  - `audio_processor.py`: The ear of the system, capturing audio via C-Level DLL bindings.
  - `ml_engine.py` & `librosa_engine.py`: The brain. Manages ML predictions, Q-Factor math, and smoothing algorithms.
  - `db_manager.py`: The memory. Handles SQLite database operations for user presets.
  - `eq_controller.py`: The hand. Translates calculations into Equalizer APO configuration parameters.
- **`/views` (The Frontend):** Streamlit modules serving the Real-Time EQ dashboard, Manual EQ controls, and Smart Playlist interfaces.

---

## 🛠️ Installation & Setup

### ⚠️ Prerequisites
1. **Python 3.11** (Strictly recommended. Versions 3.12/3.13 may face Windows DLL compatibility issues with audio libraries).
2. [**Equalizer APO**](https://sourceforge.net/projects/equalizerapo/) installed and configured on your primary audio device.

### ⚙️ Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/BButcher-435/Drupe.git](https://github.com/BButcher-435/Drupe.git)
cd Drupe

# 2. Set up a virtual environment
python -3.11 -m venv venv

# 3. Activate the environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the application
streamlit run app.py

```
### Contributing
Contributions are always welcome! Feel free to open an issue or submit a pull request if you have ideas to improve the DSP pipeline, UI, or ML models
