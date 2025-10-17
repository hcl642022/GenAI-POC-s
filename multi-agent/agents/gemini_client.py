import os
import google.generativeai as genai
from dotenv import load_dotenv
 
# Load environment variables from .env file
load_dotenv()
 
# Fetch API key and model from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
 
# Configure Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)
 
 
def call_gemini(prompt: str, max_output_tokens: int = 512) -> str:
    """
    Calls the Gemini model with the given prompt.
    Returns generated text response.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_output_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            }
        )
 
        return response.text
 
    except Exception as e:
        return f"[Gemini API Error] {str(e)}"