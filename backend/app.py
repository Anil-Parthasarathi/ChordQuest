from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
from pathlib import Path
from search import bm25_Search

# Load environment variables from root .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Access API keys (example - use when needed)
# LASTFM_API_KEY = os.getenv('LASTFM_API_KEY')
# SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Flask backend is running'})

@app.route('/api/hello', methods=['GET'])
def hello():
    """Sample API endpoint"""
    return jsonify({'message': 'Hello from Flask!'})

@app.route('/api/search', methods=['GET'])
def search_sheet_music():
    """Search endpoint"""
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({'error': 'missing query parameter'}), 400

    result = bm25_Search(query)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
