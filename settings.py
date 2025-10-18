
MODEL_NAME = "gpt-4o-mini"   # small, fast, cheap model
USE_OPENAI = True
CSV_PATH = "course.csv"
CSV_PATH = r"C:\Users\themi\Documents\Projects\chatbot\course.csv"
INDEX_DIR = "vs_courses" # FAISS index

# Output behavior
TOP_K = 3 # retrieval depth
TEMPERATURE = 0.2 # calmer = less rambling
NUM_PREDICT = 512 # cap output length
