MODEL_NAME = "deepseek-r1:8b" #OLLAMA model name
CSV_PATH = r"C:\Users\themi\Desktop\Uni projects\Software engineer project\Data set\Curriculums\CS\course.csv"
INDEX_DIR = "vs_courses"      # FAISS index
TOP_K = 10                # retrival depth

# Output behavior
USE_JSON     = True               # force JSON-only answers (no “thinking”)
TEMPERATURE  = 0.2                # calmer = less rambling
NUM_PREDICT  = 256                # cap output length