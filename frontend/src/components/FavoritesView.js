import React from 'react';

function FavoritesView({ 
  favoriteSongs, 
  setCurrentView, 
  handleSongClick, 
  toggleFavorite, 
  isFavorite 
}) {
  return (
    <div className="favorites-view">
      <div className="results-header">
        <h2>My Favorites</h2>
        <button className="back-btn" onClick={() => setCurrentView('home')}>
          ← Back to Search
        </button>
      </div>
      <div className="song-list">
        {favoriteSongs.length > 0 ? (
          favoriteSongs.map(song => (
            <div 
              key={song.id} 
              className="song-item"
              onClick={() => handleSongClick(song.id)}
            >
              <div className="song-info">
                <h3 className="song-title">{song.title}</h3>
                <p className="song-artist">{song.artist}</p>
              </div>
              <div className="song-details">
                <span className="song-difficulty">{song.difficulty}</span>
                <span className="song-key">{song.key}</span>
                <button 
                  className={`favorite-star ${isFavorite(song.id) ? 'favorited' : ''}`}
                  onClick={(e) => toggleFavorite(e, song.id)}
                  aria-label="Toggle favorite"
                >
                  {isFavorite(song.id) ? '★' : '☆'}
                </button>
              </div>
            </div>
          ))
        ) : (
          <p className="no-results">You haven't added any favorites yet.</p>
        )}
      </div>
    </div>
  );
}

export default FavoritesView;
