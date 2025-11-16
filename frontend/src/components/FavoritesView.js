import React from 'react';
import SongList from './SongList';

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
        <h2>My Favorites ({favoriteSongs.length} total)</h2>
        <button className="back-btn" onClick={() => setCurrentView('home')}>
          ← Back to Search
        </button>
      </div>
      
      {favoriteSongs.length > 0 ? (
        <SongList
          songs={favoriteSongs}
          handleSongClick={handleSongClick}
          toggleFavorite={toggleFavorite}
          isFavorite={isFavorite}
        />
      ) : (
        <div className="song-list">
          <p className="no-results">You haven't added any favorites yet.</p>
        </div>
      )}
    </div>
  );
}

export default FavoritesView;
