from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from search import bm25_Search
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
    
@app.route('/api/check-availability', methods=['POST'])
def check_availability():
    """Check if multiple MuseScore sheets are still available"""
    try:
        songs = request.get_json()
        if not songs:
            return jsonify({'error': 'No songs provided'}), 400
        
        def check_single_url(song):
            if not song.get('url'):
                return {'id': song['id'], 'available': False}
            
            try:
                url = song['url']
                
                # Handle URLs that already contain the full domain
                if url.startswith('https://musescore.com') or url.startswith('http://musescore.com'):
                    embed_url = f"{url}/embed"
                elif url.startswith('http'):
                    # URL has different domain or malformed (like https://musescore.comhttps://)
                    # Filter these out immediately
                    return {'id': song['id'], 'available': False}
                else:
                    # Relative path starting with /
                    embed_url = f"https://musescore.com{url}/embed"
                
                # Add headers to mimic a real browser to avoid Cloudflare blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://musescore.com/',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'iframe',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'same-origin',
                }
                
                response = requests.get(embed_url, headers=headers, timeout=5, allow_redirects=True)
                
                # Available music has about me and response code is not 404
                if "<div class=\"ms-footer--title\">About MuseScore</div>" in response.text or response.status_code == 404:  
                    return {'id': song['id'], 'available': False}
                
                return {'id': song['id'], 'available': True}
                
            except requests.exceptions.Timeout:
                # On timeout, assume it's available (to avoid false negatives)
                return {'id': song['id'], 'available': True}
            except Exception as e:
                # On other errors, assume available to avoid filtering out valid sheets
                print(f"Error checking {song.get('url')}: {e}")
                return {'id': song['id'], 'available': True}
        
        # Check multiple URLs concurrently for better performance
        availability_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers since we're using GET
            futures = {executor.submit(check_single_url, song): song for song in songs}
            for future in as_completed(futures):
                result = future.result()
                availability_map[result['id']] = result['available']
        
        return jsonify({'availability': availability_map})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
