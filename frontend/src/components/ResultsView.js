import React from 'react';

function ResultsView({ 
  searchResults, 
  setCurrentView, 
  handleSongClick, 
  toggleFavorite, 
  isFavorite,
  currentPage,
  setCurrentPage
}) {
  const RESULTS_PER_PAGE = 20;
  const totalPages = Math.ceil(searchResults.length / RESULTS_PER_PAGE);
  const startIndex = (currentPage - 1) * RESULTS_PER_PAGE;
  const endIndex = startIndex + RESULTS_PER_PAGE;
  const currentResults = searchResults.slice(startIndex, endIndex);

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="results-view">
      <div className="results-header">
        <h2>Search Results ({searchResults.length} total)</h2>
        <button className="back-btn" onClick={() => setCurrentView('home')}>
          ← Back to Search
        </button>
      </div>
      
      <div className="song-list">
        {currentResults.length > 0 ? (
          currentResults.map(song => (
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
          <p className="no-results">No songs found matching your search.</p>
        )}
      </div>
      
      {totalPages > 1 && (
        <div className="pagination">
          <button 
            className="pagination-btn"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
          >
            ← Previous
          </button>
          <div className="pagination-info">
            <span className="page-numbers">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }
                return (
                  <button
                    key={pageNum}
                    className={`page-number ${currentPage === pageNum ? 'active' : ''}`}
                    onClick={() => handlePageChange(pageNum)}
                  >
                    {pageNum}
                  </button>
                );
              })}
            </span>
            <span className="page-text">Page {currentPage} of {totalPages}</span>
          </div>
          <button 
            className="pagination-btn"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}

export default ResultsView;
