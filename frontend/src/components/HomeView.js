import React from 'react';
import RecommendedSongsList from './RecommendedSongsList';

function HomeView({ 
  searchQuery, 
  setSearchQuery, 
  searchType,
  setSearchType,
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
        <div className="search-controls">
          <div className="search-type-toggle">
            <button
              type="button"
              className={`toggle-btn ${searchType === 'bm25' ? 'active' : ''}`}
              onClick={() => setSearchType('bm25')}
            >
              BM25
            </button>
            <button
              type="button"
              className={`toggle-btn ${searchType === 'embedding' ? 'active' : ''}`}
              onClick={() => setSearchType('embedding')}
            >
              Embedding
            </button>
          </div>
        </div>
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
      <RecommendedSongsList
        songs={recommendedSongs}
        handleSongClick={handleSongClick}
        toggleFavorite={toggleFavorite}
        isFavorite={isFavorite}
        title="Recommended for You"
      />
    </div>
  );
}

export default HomeView;
