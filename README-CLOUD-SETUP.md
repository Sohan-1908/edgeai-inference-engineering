# EdgeAI cloud demo setup

Copy `cloud_app.py` and `requirements-cloud.txt` into the root of your EdgeAI repository. Keep the existing `requirements.txt` unchanged. The app expects `models/yolo11n.onnx`.

Local test (PowerShell):
```powershell
py -3.14 -m venv .venv-cloud
.\.venv-cloud\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cloud.txt
streamlit run cloud_app.py
```
If ONNX Runtime has no compatible wheel for your local Python version, use Python 3.12 or 3.13. In Streamlit Community Cloud, choose the Python version in Advanced settings. This is a CPU demo, not a GPU benchmark.
