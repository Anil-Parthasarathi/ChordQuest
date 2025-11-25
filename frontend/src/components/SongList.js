import React from 'react';

function SongList({ 
  songs, 
  handleSongClick, 
  toggleFavorite, 
  isFavorite 
}) {
  return (
    <div className="song-list">
      {songs.length > 0 ? (
        songs.map(song => (
          <div 
            key={song.id} 
            className="song-item"
            onClick={() => handleSongClick(song.id)}
          >
            <div className="song-info">
              <div className="song-title-row">
                <h3 className="song-title">{song.title}</h3>
                {song.instrumentsNames && song.instrumentsNames.length > 0 && (
                  <div className="song-instruments-badges">
                    {song.instrumentsNames.slice(0, 4).map((instrument, idx) => (
                      <span key={idx} className="instrument-badge">{instrument}</span>
                    ))}
                    {song.instrumentsNames.length > 4 && (
                      <span className="instrument-badge instrument-badge-more">+{song.instrumentsNames.length - 4}</span>
                    )}
                  </div>
                )}
              </div>
              <div className="song-meta-row">
                {song.timeUpdated && (
                  <p className="song-updated">{new Date(song.timeUpdated).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                )}
                <div className="song-meta-tags">
                  {song.duration > 0 && <span className="meta-tag meta-tag-duration">⏱ {Math.floor(song.duration / 60)}:{String(song.duration % 60).padStart(2, '0')}</span>}
                  {song.pagesCount > 0 && <span className="meta-tag">{song.pagesCount} pg</span>}
                  {song.partsCount > 0 && <span className="meta-tag">{song.partsCount} parts</span>}
                </div>
              </div>
            </div>
            <div className="song-right-section">
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
        <p className="no-results">No songs found.</p>
      )}
    </div>
  );
}

export default SongList;
