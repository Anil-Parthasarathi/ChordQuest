import React from 'react';

function RecommendedSongsList({ 
  songs, 
  handleSongClick, 
  toggleFavorite, 
  isFavorite,
  title = "Recommended for You",
  variant = "recommendation" // "recommendation" or "similar"
}) {
  const sectionClass = variant === "similar" ? "similar-songs-section" : "recommendations-section";
  const listClass = variant === "similar" ? "similar-songs-list" : "recommendations-list";
  const cardClass = variant === "similar" ? "similar-song-card" : "recommendation-card";
  const contentClass = variant === "similar" ? "similar-song-info" : "recommendation-content";
  const titleRowClass = variant === "similar" ? "similar-title-row" : "recommendation-title-row";
  const songTitleClass = variant === "similar" ? "similar-song-title" : "recommendation-song-title";
  const instrumentsBadgesClass = variant === "similar" ? "similar-instruments-badges" : "recommendation-instruments-badges";
  const instrumentBadgeClass = variant === "similar" ? "instrument-badge-similar" : "instrument-badge-rec";
  const artistClass = variant === "similar" ? "similar-song-artist" : "recommendation-artist";
  const tagsClass = variant === "similar" ? "similar-song-tags" : "recommendation-tags";
  const tagClass = variant === "similar" ? "similar-tag" : "recommendation-tag";
  const durationTagClass = variant === "similar" ? "similar-tag similar-tag-duration" : "recommendation-tag recommendation-tag-duration";
  const metaRowClass = variant === "similar" ? "similar-meta-row" : null;
  const titleClass = variant === "similar" ? "section-title" : "recommendations-title";

  return (
    <div className={sectionClass}>
      <h2 className={titleClass}>{title}</h2>
      <div className={listClass}>
        {songs.map(song => (
          <div 
            key={song.id} 
            className={cardClass}
            onClick={() => handleSongClick(song.id)}
          >
            <div className={contentClass}>
              {variant === "similar" ? (
                <>
                  <div className={titleRowClass}>
                    <h3 className={songTitleClass}>{song.title}</h3>
                  </div>
                  <div className={metaRowClass}>
                    {song.timeUpdated && (
                      <p className={artistClass}>Updated {new Date(song.timeUpdated).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                    )}
                    {song.instrumentsNames && song.instrumentsNames.length > 0 && (
                      <div className={instrumentsBadgesClass}>
                        {song.instrumentsNames.slice(0, 2).map((instrument, idx) => (
                          <span key={idx} className={instrumentBadgeClass}>{instrument}</span>
                        ))}
                        {song.instrumentsNames.length > 2 && (
                          <span className={`${instrumentBadgeClass} instrument-badge-more`}>+{song.instrumentsNames.length - 2}</span>
                        )}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className={titleRowClass}>
                    <h3 className={songTitleClass}>{song.title}</h3>
                  </div>
                  {song.instrumentsNames && song.instrumentsNames.length > 0 ? (
                    <div className={instrumentsBadgesClass}>
                      {song.instrumentsNames.slice(0, 2).map((instrument, idx) => (
                        <span key={idx} className={instrumentBadgeClass}>{instrument}</span>
                      ))}
                      {song.instrumentsNames.length > 2 && (
                        <span className={`${instrumentBadgeClass} instrument-badge-more`}>+{song.instrumentsNames.length - 2}</span>
                      )}
                    </div>
                  ) : (
                    <div style={{ height: '28px' }}></div>
                  )}
                  {song.timeUpdated && (
                    <p className={artistClass}>Updated {new Date(song.timeUpdated).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                  )}
                </>
              )}
              <div className={tagsClass}>
                {song.duration > 0 && <span className={durationTagClass}>⏱ {Math.floor(song.duration / 60)}:{String(song.duration % 60).padStart(2, '0')}</span>}
                {song.pagesCount > 0 && <span className={tagClass}>{song.pagesCount} pg</span>}
                {song.partsCount > 0 && <span className={tagClass}>{song.partsCount} parts</span>}
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
  );
}

export default RecommendedSongsList;
