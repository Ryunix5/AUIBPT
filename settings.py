# settings.py
MODEL_NAME = "groq" 
USE_OPENAI = True            # set False locally if you want to use Ollama

# Data / index
CSV_PATH = "course.csv"      # keep relative for Streamlit Cloud
INDEX_DIR = "vs_courses"     # FAISS index folder

# Output behavior
TOP_K = 3
TEMPERATURE = 0.1
NUM_PREDICT = 4096

USE_GROQ_ONLY = True