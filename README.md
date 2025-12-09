# ChordQuest

The new go-to source for finding music sheets!

ChordQuest is a search engine and recommender system that helps you discover your next rocking performance! Search through hundreds of thousands of sheet music from MuseScore and get personalized recommendations based on your favorites.

## Features

- **Dual Search Modes**: 
  - **BM25 Search**: Traditional keyword-based search with TF-IDF scoring
  - **Semantic Embedding Search**: AI-powered search using sentence embeddings (all-MiniLM-L6-v2) for understanding query intent
  
- **Hybrid Recommendation System**:
  - **RecVAE**: Variational autoencoder trained on ~84K popular works for collaborative filtering
  - **BPR (Bayesian Personalized Ranking)**: Item embeddings covering ~612K works for similarity-based recommendations
  - Automatic fallback and blending between models based on item coverage

- **Favorites Management**: Save your favorite sheets and get personalized recommendations
- **Sheet Music Preview**: Embedded MuseScore viewer to preview sheets directly in the app
- **Similar Songs**: Get recommendations for similar songs when viewing any sheet

## Project Structure

```
ChordQuest/
├── backend/                    # Flask backend
│   ├── app.py                  # Main Flask API server
│   ├── search.py               # BM25 search implementation
│   ├── search_embedding.py     # Semantic search with FAISS
│   ├── recbole_recommender.py  # Hybrid RecVAE/BPR recommender
│   ├── metadata_loader.py      # Sheet metadata management
│   ├── requirements.txt
│   ├── api_lookups/            # Sheet-to-work ID mappings
│   ├── models/                 # Trained model weights and training files
│   ├── musescore_metadata/     # Song metadata and indices
│   └── scraping_tools/         # Data collection utilities
├── frontend/                   # React frontend
│   ├── public/
│   ├── src/
│   │   ├── App.js              # Main application component
│   │   └── components/
│   │       ├── HomeView.js         # Home page with search
│   │       ├── ResultsView.js      # Search results display
│   │       ├── FavoritesView.js    # Favorites management
│   │       ├── SongDetailView.js   # Sheet detail with embed
│   │       ├── SongList.js         # Reusable song list
│   │       ├── RecommendedSongsList.js
│   │       └── Navbar.js
│   └── package.json
└── README.md
```

## Setup Instructions

### Backend Setup (Flask)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Flask server:
   ```bash
   python app.py
   ```

   The backend will start on `http://localhost:5000`

### Frontend Setup (React)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The frontend will start on `http://localhost:3000`

## Running the Application

1. Start the Flask backend (in one terminal):
   ```bash
   cd backend
   source venv/bin/activate
   python app.py
   ```

2. Start the React frontend (in another terminal):
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser to `http://localhost:3000` to see the app running

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check endpoint |
| `/api/search` | GET | Search for sheet music (params: `query`, `type=bm25\|embedding`) |
| `/api/setfavorite` | POST | Add a song to favorites |
| `/api/removeFavorite` | POST | Remove a song from favorites |
| `/api/retrieveFavorites` | GET | Get all favorited songs |
| `/api/recommendations` | POST | Get song recommendations (body: `{songIds: [], k: 10}`) |
| `/api/check-availability` | POST | Verify sheets are still available on MuseScore |

## Tech Stack

### Frontend
- **React.js 18.2** - UI framework
- **CSS3** - Styling

### Backend
- **Python Flask 3.0** - Web framework
- **CORS** - flask-cors for cross-origin requests
- **Sentence Transformers** - Semantic embeddings (all-MiniLM-L6-v2)
- **FAISS** - Vector similarity search
- **PyTorch** - Deep learning models
- **RecBole** - Recommendation library
- **Pandas/NumPy** - Data processing

### Machine Learning Models
- **RecVAE** - Variational autoencoder for collaborative filtering
- **BPR** - Bayesian Personalized Ranking for item embeddings
- **all-MiniLM-L6-v2** - Sentence transformer for semantic search

## Data Sources

- **MuseScore** - Sheet music metadata and embeds
- **Last.fm** - Music interaction data for training recommendation models
