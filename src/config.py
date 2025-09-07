import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        # API Key
        self.GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if not self.GOOGLE_AI_STUDIO_API_KEY:
            raise ValueError("Please set GOOGLE_AI_STUDIO_API_KEY in your .env file")

        # Model configuration
        self.GOOGLE_MODEL = "/gemini-2.0-flash:"
        self.GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        # Path configuration
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.ARTIFACTS_DIR = os.path.join(self.BASE_DIR, "artifacts")
        self.TEST_DATA_DIR = os.path.join(self.BASE_DIR, "test_data")

        # API configuration
        self.MAX_RETRIES = 3
        self.RETRY_DELAY_BASE = 2
        self.CHUNK_SIZE = 4000

        # Validation thresholds
        self.VALIDATION_WARNING_THRESHOLD = 0.1  # 10% variation allowed
        self.MAX_HALLUCINATION_WARNINGS = 5

        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.TEST_DATA_DIR, exist_ok=True)

config = Config()