import os

from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient

load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_INDEX_NAME = "vector_index"
API_KEY = os.environ.get("API_KEY")

client = genai.Client()

mongo = MongoClient(os.environ["MONGODB_URI"])
db = mongo["carelens_py"]
chunks_collection = db["report_chunks"]