from app import app
from waitress import serve
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    try:
        serve(app, host='127.0.0.1', port=8000)
    except Exception as e:
        print(f"Error starting server: {e}")
