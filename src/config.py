import os                              
from dotenv import load_dotenv   

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "❌ GEMINI_API_KEY missing. Add it in .env at project root like: GEMINI_API_KEY=your_key_here"
    )