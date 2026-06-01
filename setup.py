from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="equai", # PyPI'da başkası almadıysa bu isimle yayınlanacak. Alındıysa equai-dsp gibi bir şey yap.
    version="1.0.0",
    author="BButcher",
    author_email="bur4q3@gmail.com",
    description="A real-time Machine Learning audio engine for dynamic Equalizer APO optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BButcher-435/Drupe",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires="==3.11.*",
    install_requires=[
        "streamlit==1.32.0"
        "spotipy==2.23.0"
"pandas==2.2.1"
"scikit-learn==1.4.1.post1"
"python-dotenv==1.0.1"
"librosa>=0.10.1"
"soundcard>=0.4.2"
"pywin32>=306"
"sounddevice>=0.4.6"
"requests>=2.31.0"
    ],
)
