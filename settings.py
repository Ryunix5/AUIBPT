MODEL_NAME = "llama3.1:8b"
CSV_PATH = r"C:\Users\themi\Documents\Projects\chatbot\courses.csv"
INDEX_DIR = "vs_courses"      # FAISS index
TOP_K = 3                # retrieval depth

# Output behavior
USE_JSON     = False               # force JSON-only answers (no “thinking”)
TEMPERATURE  = 0.2                # calmer = less rambling
NUM_PREDICT  = 512                # cap output length
