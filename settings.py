# settings.py
MODEL_NAME = "gpt-4o-mini"   # fast, low-cost OpenAI model
USE_OPENAI = True            # set False locally if you want to use Ollama

# Data / index
CSV_PATH = "course.csv"      # keep relative for Streamlit Cloud
INDEX_DIR = "vs_courses"     # FAISS index folder

# Output behavior
TOP_K = 3
TEMPERATURE = 0.2
NUM_PREDICT = 256            # used as max_tokens for OpenAI
