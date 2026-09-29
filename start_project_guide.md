cd C:\Users\Himanshu\Downloads\student-rag-chat
.venv\Scripts\Activate.ps1
agar folder path cange kiya hai .venv delete karke chalu karna hoga
deactivate
Remove-Item -Recurse -Force .venv
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv run pytest -v
uv run streamlit run app.py
