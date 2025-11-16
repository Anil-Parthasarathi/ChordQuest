import React from 'react';

function HomeView({ 
  searchQuery, 
  setSearchQuery, 
  handleSearch, 
  recommendedSongs, 
  handleSongClick, 
  toggleFavorite, 
  isFavorite 
}) {
  return (
    <div className="home-view">
      <div className="hero-section">
        <img
          src={process.env.PUBLIC_URL + '/assets/chordQuestFavicon.png'}
          alt="ChordQuest"
          className="hero-logo"
        />
        <h2 className="hero-title">Discover Your Next Song</h2>
        <p className="hero-subtitle">Search millions of music sheets and chords</p>
      </div>

      <form onSubmit={handleSearch} className="search-form">
        <div className="search-container">
          <input
            type="text"
            className="search-input"
            placeholder="Search for songs, artists, or genres..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button type="submit" className="search-btn">
            Search
          </button>
        </div>
      </form>

      {/* Recommendations Section */}
      <div className="recommendations-section">
        <h2 className="recommendations-title">Recommended for You</h2>
        <div className="recommendations-list">
          {recommendedSongs.map(song => (
            <div 
              key={song.id} 
              className="recommendation-card"
              onClick={() => handleSongClick(song.id)}
            >
              <div className="recommendation-content">
                <div className="recommendation-title-row">
                  <h3 className="recommendation-song-title">{song.title}</h3>
                  {song.instrumentsNames && song.instrumentsNames.length > 0 && (
                    <div className="recommendation-instruments-badges">
                      {song.instrumentsNames.slice(0, 2).map((instrument, idx) => (
                        <span key={idx} className="instrument-badge-rec">{instrument}</span>
                      ))}
                      {song.instrumentsNames.length > 2 && (
                        <span className="instrument-badge-rec instrument-badge-more">+{song.instrumentsNames.length - 2}</span>
                      )}
                    </div>
                  )}
                </div>
                {song.timeUpdated && (
                  <p className="recommendation-artist">Updated {new Date(song.timeUpdated).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                )}
                <div className="recommendation-tags">
                  {song.duration > 0 && <span className="recommendation-tag recommendation-tag-duration">⏱ {Math.floor(song.duration / 60)}:{String(song.duration % 60).padStart(2, '0')}</span>}
                  {song.pagesCount > 0 && <span className="recommendation-tag">{song.pagesCount} pg</span>}
                  {song.partsCount > 0 && <span className="recommendation-tag">{song.partsCount} parts</span>}
                </div>
              </div>
              <button 
                className={`favorite-star ${isFavorite(song.id) ? 'favorited' : ''}`}
                onClick={(e) => toggleFavorite(e, song.id)}
                aria-label="Toggle favorite"
              >
                {isFavorite(song.id) ? '★' : '☆'}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default HomeView;
