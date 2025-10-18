# 🎓 AUIBPT — AUIB Course & Faculty Chatbot / Planner

**AUIBPT** is an intelligent chatbot and schedule-builder for the  
**American University of Iraq – Baghdad (AUIB)** Colleges of Arts & Sciences (CAS),  
Pharmacy (COP), and Dentistry (COD).

Built by **Ryunix Productions**, it combines:
- 🧠 Institutional knowledge (university info + faculty profiles)
- 📚 Course catalog lookup and prerequisite logic
- 🗓️ Smart semester planner with liberal-arts rules
- 🎨 Profile picture + color customizer GUI
- 💬 Chat assistant (personalized by completed courses)

---

## 🛠 Setup

### 1️⃣ Clone or download
```bash
git clone https://github.com/yourname/auibpt.git
cd auibpt
2️⃣ Create virtual environment
bash
Copy code
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Choose your backend
🟢 Option A — OpenAI API (Cloud)
Create account → API key

Set your key:

bash
Copy code
export OPENAI_API_KEY="sk-your-key-here"
Edit settings.py:

python
Copy code
MODEL_NAME = "gpt-4o-mini"
USE_OPENAI = True
⚙️ Option B — Local Ollama (Offline)
Install Ollama

Pull the model:

bash
Copy code
ollama pull llama3.1:8b
In settings.py:

python
Copy code
MODEL_NAME = "llama3.1:8b"
USE_OPENAI = False
▶️ Run
bash
Copy code
streamlit run app.py
Local URL: http://localhost:8501
(You can safely close the terminal after pressing Ctrl +C.)

🧩 Files Overview
File	Purpose
app.py	Main Streamlit interface + AI logic
settings.py	Model and index configuration
data_loader.py / indexer.py	CSV & FAISS index helpers
course.csv	Master catalog (CAS/COP/COD courses)
auib_university_kb.json	Institutional knowledge (facts + faculty)
RP.png	Ryunix logo / app icon
requirements.txt	Python dependencies

💬 Features
Chatbot: Ask about courses, professors, or AUIB itself.

Planner: Select completed courses → AI suggests next semester.

Swap & Lock: Replace or lock courses within schedule.

Progress: Tracks credits toward your degree (126 CS / 180 Pharm / 189 Dentistry).

Profile Avatar + Color Customizer.

Save/Load Schedules (.json) and Export to CSV.

☁️ Deploying to Streamlit Cloud
Push your repo to GitHub.

Go to https://share.streamlit.io.

Connect your repo → select app.py.

Add a secrets entry:

bash
Copy code
OPENAI_API_KEY="sk-your-key-here"
Deploy → share the URL (https://auibpt.streamlit.app).

🧑‍🏫 Credits
Developed by Ryunix Productions.
