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


for Mac/Linux
cd ~/Projects/chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install Ollama (https://ollama.com/download) and pull model:
ollama pull llama3:instruct
streamlit run app.py
