import React from 'react';

function SongDetailView({ 
  song, 
  setCurrentView,
  previousView = 'home', // Default to home if not provided
  toggleFavorite, 
  isFavorite,
  handleSongClick,
  recommendedSongs 
}) {
  if (!song) {
    return (
      <div className="song-detail-view">
        <div className="detail-header">
          <button className="back-btn" onClick={() => setCurrentView(previousView)}>
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
        <button className="back-btn" onClick={() => setCurrentView(previousView)}>
          ← Back to {previousView === 'results' ? 'Search Results' : previousView === 'favorites' ? 'Favorites' : 'Home'}
        </button>
      </div>

      {/* Song Information Section */}
      <div className="song-info-section">
        <div className="song-header-content">
          <div className="song-main-info">
            <div className="detail-title-row">
              <h1 className="detail-song-title">{song.title}</h1>
              {song.instrumentsNames && song.instrumentsNames.length > 0 && (
                <div className="detail-instruments-badges">
                  {song.instrumentsNames.slice(0, 3).map((instrument, idx) => (
                    <span key={idx} className="instrument-badge-detail-inline">{instrument}</span>
                  ))}
                  {song.instrumentsNames.length > 3 && (
                    <span className="instrument-badge-detail-inline instrument-badge-more">+{song.instrumentsNames.length - 3}</span>
                  )}
                </div>
              )}
            </div>
            <div className="detail-song-meta">
              {song.timeUpdated && (
                <span className="detail-song-artist">{new Date(song.timeUpdated).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
              )}
              {song.duration > 0 && (
                <>
                  {song.timeUpdated && <span className="meta-divider">•</span>}
                  <span className="detail-song-duration">⏱ {Math.floor(song.duration / 60)}:{String(song.duration % 60).padStart(2, '0')}</span>
                </>
              )}
            </div>
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
          {song.pagesCount > 0 && (
            <div className="metadata-item metadata-item-classy">
              <span className="metadata-icon">📄</span>
              <div className="metadata-content">
                <span className="metadata-label">Pages</span>
                <span className="metadata-value">{song.pagesCount}</span>
              </div>
            </div>
          )}
          {song.partsCount > 0 && (
            <div className="metadata-item metadata-item-classy">
              <span className="metadata-icon">🎼</span>
              <div className="metadata-content">
                <span className="metadata-label">Parts</span>
                <span className="metadata-value">{song.partsCount}</span>
              </div>
            </div>
          )}
          {song.partsNames && song.partsNames.length > 0 && (
            <div className="metadata-item metadata-item-classy metadata-item-wide">
              <span className="metadata-icon">🎹</span>
              <div className="metadata-content">
                <span className="metadata-label">Part Names</span>
                <span className="metadata-value">{song.partsNames.join(', ')}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chords Display Section */}
      <div className="chords-section">
        <h2 className="section-title">Music</h2>
        <div className="chords-container">
          {song.url && (
            <>
              <iframe 
                id="score-iframe" 
                width="100%" 
                height="1000" 
                src={`https://musescore.com${song.url}/embed`} 
                frameborder="0" 
                allowfullscreen 
                allow="autoplay; fullscreen"
                title="Sheet Music"
              >
              </iframe>
              <div className="sheet-music-credit">
                <a href={`https://musescore.com${song.url}`} target="_blank" rel="noreferrer" className="sheet-music-title">{song.title}</a> by <a href={`https://musescore.com/user/${song.authorUserId || song.artist}`} target="_blank" rel="noreferrer" className="sheet-music-author">{song.artist}</a> via MuseScore
              </div>
            </>
          )}
          {!song.url && (
            <p className="no-results">Sheet music embed not available.</p>
          )}
        </div>
      </div>

      {/* Description Section */}
      {song.description && (
        <div className="description-section">
          <h2 className="section-title">Description</h2>
          <p className="song-description">{song.description}</p>
        </div>
      )}

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
                <div className="similar-title-row">
                  <h3 className="similar-song-title">{recSong.title}</h3>
                </div>
                <div className="similar-meta-row">
                  {recSong.timeUpdated && (
                    <p className="similar-song-artist">{new Date(recSong.timeUpdated).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                  )}
                  {recSong.instrumentsNames && recSong.instrumentsNames.length > 0 && (
                    <div className="similar-instruments-badges">
                      {recSong.instrumentsNames.slice(0, 2).map((instrument, idx) => (
                        <span key={idx} className="instrument-badge-similar">{instrument}</span>
                      ))}
                      {recSong.instrumentsNames.length > 2 && (
                        <span className="instrument-badge-similar instrument-badge-more">+{recSong.instrumentsNames.length - 2}</span>
                      )}
                    </div>
                  )}
                </div>
                <div className="similar-song-tags">
                  {recSong.duration > 0 && <span className="similar-tag similar-tag-duration">⏱ {Math.floor(recSong.duration / 60)}:{String(recSong.duration % 60).padStart(2, '0')}</span>}
                  {recSong.pagesCount > 0 && <span className="similar-tag">{recSong.pagesCount} pg</span>}
                  {recSong.partsCount > 0 && <span className="similar-tag">{recSong.partsCount} parts</span>}
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
