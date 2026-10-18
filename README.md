# SindhiNet Streamlit GUI

Keep these files together: `app.py`, `best_sindhi_ocr_model.keras`, `class_names.json`, and `requirements.txt`.

## Run in Windows / VS Code

Open this folder in VS Code, then Terminal → New Terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

The app uses grayscale 64x64 images and scales pixels to [0, 1]. Confirm the label order in `class_names.json` matches the order used to train the model.
