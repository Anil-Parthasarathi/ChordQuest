import React from 'react';

function SongDetailView({ 
  song, 
  setCurrentView, 
  toggleFavorite, 
  isFavorite,
  handleSongClick,
  recommendedSongs 
}) {
  if (!song) {
    return (
      <div className="song-detail-view">
        <div className="detail-header">
          <button className="back-btn" onClick={() => setCurrentView('home')}>
            ← Back
          </button>
        </div>
        <p className="no-results">Song not found.</p>
      </div>
    );
  }

  return (
    <div className="song-detail-view">
      {/* Header with back button */}
      <div className="detail-header">
        <button className="back-btn" onClick={() => setCurrentView('home')}>
          ← Back to Search
        </button>
      </div>

      {/* Song Information Section */}
      <div className="song-info-section">
        <div className="song-header-content">
          <div className="song-main-info">
            <h1 className="detail-song-title">{song.title}</h1>
            <p className="detail-song-artist">by {song.artist}</p>
          </div>
          <button 
            className={`favorite-star-large ${isFavorite(song.id) ? 'favorited' : ''}`}
            onClick={(e) => toggleFavorite(e, song.id)}
            aria-label="Toggle favorite"
          >
            {isFavorite(song.id) ? '★' : '☆'}
          </button>
        </div>
        
        <div className="song-metadata">
          <div className="metadata-item">
            <span className="metadata-label">Difficulty:</span>
            <span className="metadata-value difficulty-badge">{song.difficulty}</span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Key:</span>
            <span className="metadata-value key-badge">{song.key}</span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Tempo:</span>
            <span className="metadata-value">{song.tempo || '120 BPM'}</span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Time Signature:</span>
            <span className="metadata-value">{song.timeSignature || '4/4'}</span>
          </div>
        </div>
      </div>

      {/* Chords Display Section */}
      <div className="chords-section">
        <h2 className="section-title">Chords & Lyrics</h2>
        <div className="chords-container">
          <iframe id="score-iframe" width="100%" height="394" src="https://musescore.com/user/27224589/scores/4978804/embed" frameborder="0" allowfullscreen allow="autoplay; fullscreen">
          </iframe>
          <div className="sheet-music-credit">
            <a href="https://musescore.com/user/27224589/scores/4978804" target="_blank" className="sheet-music-title">Viva La Vida Guitar and Violin</a> by <a href="https://musescore.com/user/27224589" target="_blank" className="sheet-music-author">kožo92</a> via MuseScore
          </div>
        </div>``
      </div>

      {/* Similar Songs Recommendations */}
      <div className="similar-songs-section">
        <h2 className="section-title">Similar Songs</h2>
        <div className="similar-songs-list">
          {recommendedSongs.map(recSong => (
            <div 
              key={recSong.id} 
              className="similar-song-card"
              onClick={() => handleSongClick(recSong.id)}
            >
              <div className="similar-song-info">
                <h3 className="similar-song-title">{recSong.title}</h3>
                <p className="similar-song-artist">{recSong.artist}</p>
                <div className="similar-song-tags">
                  <span className="similar-tag">{recSong.difficulty}</span>
                  <span className="similar-tag">{recSong.key}</span>
                </div>
              </div>
              <button 
                className={`favorite-star ${isFavorite(recSong.id) ? 'favorited' : ''}`}
                onClick={(e) => toggleFavorite(e, recSong.id)}
                aria-label="Toggle favorite"
              >
                {isFavorite(recSong.id) ? '★' : '☆'}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default SongDetailView;
