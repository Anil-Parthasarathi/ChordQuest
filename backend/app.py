from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from search import bm25_Search

# Load environment variables from root .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

favorites_file = Path(__file__).parent / 'user_favorites.json'

def load_favorites():
    """Load favorites from JSON file"""
    if favorites_file.exists() and favorites_file.stat().st_size > 0:
        try:
            with open(favorites_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_favorites(favorites):
    """Save favorites to JSON file"""
    with open(favorites_file, 'w', encoding='utf-8') as f:
        json.dump(favorites, f, indent=2, ensure_ascii=False)

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

@app.route('/api/setfavorite', methods=['POST'])
def set_favorite():
    """Set favorite endpoint"""
    try:
        song_data = request.get_json()
        if not song_data or 'id' not in song_data:
            return jsonify({'error': 'missing song data'}), 400
        
        favorites = load_favorites()
        
        # Check if song is already favorited
        if not any(fav['id'] == song_data['id'] for fav in favorites):
            favorites.append(song_data)
            save_favorites(favorites)
        
        return jsonify({'message': 'Song added to favorites', 'favorites': favorites})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/removeFavorite', methods=['POST'])
def remove_favorite():
    """Remove favorite endpoint"""
    try:
        data = request.get_json()
        if not data or 'id' not in data:
            return jsonify({'error': 'missing song id'}), 400
        
        song_id = data['id']
        favorites = load_favorites()
        favorites = [fav for fav in favorites if fav['id'] != song_id]
        save_favorites(favorites)
        
        return jsonify({'message': 'Song removed from favorites', 'favorites': favorites})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrieveFavorites', methods=['GET'])
def retrieve_favorites():
    """Retrieve favorites endpoint"""
    try:
        favorites = load_favorites()
        return jsonify({'favorites': favorites})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
