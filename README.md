# 1) Install Python 3.10+ if needed
# 2) Clone or unzip the project, then:
cd C:\Users\<me>\Documents\Projects\chatbot

# 3) Create a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 4) Install dependencies
pip install -r requirements.txt

# 5) Install Ollama and the model (only once)
# Download Ollama: https://ollama.com/download
# Then pull the model specified in settings.MODEL_NAME, e.g.:
ollama pull llama3:instruct

# 6) Run the app
streamlit run app.py
