import React, { useState, useEffect, useRef } from 'react';
import SongList from './SongList';

// Store filter state outside component to persist across re-renders
let savedFilterState = null;
let globalAvailabilityCache = {};

function ResultsView({ 
  searchResults, 
  setCurrentView, 
  handleSongClick, 
  toggleFavorite, 
  isFavorite,
  currentPage,
  setCurrentPage
}) {
  const RESULTS_PER_PAGE = 15;

  // Extract all unique instruments from all songs, removing numbers in parentheses
  // Normalize to title case for consistent display
  const normalizeInstrument = (instr) => {
    const cleaned = instr.replace(/\s*\(\d+\)\s*$/, '').trim();
    // Convert to title case: capitalize first letter of each word
    return cleaned.split(' ').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
  };

  const availableInstruments = Array.from(
    new Set(
      searchResults.flatMap(song => 
        (song.instrumentsNames || []).map(instr => normalizeInstrument(instr))
      )
    )
  ).filter(instr => instr).sort();

  // Calculate max values from search results
  const maxDuration = Math.min(Math.max(...searchResults.map(song => song.duration || 0)), 3600);
  const maxParts = Math.max(...searchResults.map(song => song.partsCount || 0), 1);
  const maxPages = Math.max(...searchResults.map(song => song.pagesCount || 0), 1);
  
  // Filter state - restore from saved state if available, otherwise use defaults
  const [showFilters, setShowFilters] = useState(savedFilterState?.showFilters ?? false);
  const [selectedInstrument, setSelectedInstrument] = useState(savedFilterState?.selectedInstrument ?? 'all');
  const [durationRange, setDurationRange] = useState(savedFilterState?.durationRange ?? [0, maxDuration]);
  const [partCountRange, setPartCountRange] = useState(savedFilterState?.partCountRange ?? [1, maxParts]);
  const [pageCountRange, setPageCountRange] = useState(savedFilterState?.pageCountRange ?? [1, maxPages]);

  const [availabilityMap, setAvailabilityMap] = useState(globalAvailabilityCache);
  const [isCheckingAvailability, setIsCheckingAvailability] = useState(false);
  const [hideUnavailable, setHideUnavailable] = useState(true); // Default to hiding unavailable

  // Save filter state whenever it changes
  useEffect(() => {
    savedFilterState = {
      showFilters,
      selectedInstrument,
      durationRange,
      partCountRange,
      pageCountRange
    };
  }, [showFilters, selectedInstrument, durationRange, partCountRange, pageCountRange]);

  // Wrap setCurrentView to clear filters when going home
  const handleSetCurrentView = (view) => {
    if (view === 'home') {
      savedFilterState = null;
    }
    setCurrentView(view);
  };

  const filteredResults = searchResults.filter(song => {
    const matchesInstrument = selectedInstrument === 'all' || 
      (song.instrumentsNames && song.instrumentsNames.some(instr => 
        normalizeInstrument(instr) === selectedInstrument
      ));

    const matchesDuration = song.duration >= durationRange[0] && song.duration <= durationRange[1];
    const matchesParts = (song.partsCount || 0) >= partCountRange[0] && (song.partsCount || 0) <= partCountRange[1];
    const matchesPages = (song.pagesCount || 0) >= pageCountRange[0] && (song.pagesCount || 0) <= pageCountRange[1];

    if (hideUnavailable && availabilityMap[song.id] === false) {
      return false;
    }
    
    return matchesInstrument && matchesDuration && matchesParts && matchesPages;
  });

  const totalPages = Math.ceil(filteredResults.length / RESULTS_PER_PAGE);
  const startIndex = (currentPage - 1) * RESULTS_PER_PAGE;
  const endIndex = startIndex + RESULTS_PER_PAGE;
  const currentResults = filteredResults.slice(startIndex, endIndex);

  useEffect(() => {
    const checkAvailability = async () => {
      if (currentResults.length === 0) return;
      
      // Filter out songs that are already in the cache
      const songsToCheck = currentResults.filter(song => availabilityMap[song.id] === undefined);
      
      if (songsToCheck.length === 0) {
         setIsCheckingAvailability(false);
         return;
      }
      
      setIsCheckingAvailability(true);
      try {
        const response = await fetch('/api/check-availability', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(songsToCheck)
        });
        
        if (response.ok) {
          const data = await response.json();
          // Update global cache
          Object.assign(globalAvailabilityCache, data.availability);
          setAvailabilityMap(prev => ({...prev, ...data.availability}));
        }
      } catch (err) {
        console.error('Failed to check availability:', err);
      } finally {
        setIsCheckingAvailability(false);
      }
    };
    
    // Delay slightly to avoid too many requests
    const timer = setTimeout(checkAvailability, 300);
    return () => clearTimeout(timer);
  }, [currentPage, JSON.stringify(currentResults.map(s => s.id))]);

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="results-view">
      <div className="results-header">
        <h2>Search Results ({filteredResults.length} total)</h2>
        <div className="header-buttons">
          <button 
            className="filter-toggle-btn" 
            onClick={() => setShowFilters(!showFilters)}
          >
            {showFilters ? '✕ Hide Filters' : '⚙ Show Filters'}
          </button>
          <button className="back-btn" onClick={() => handleSetCurrentView('home')}>
            ← Back to Search
          </button>
        </div>
      </div>
      
      {/* Filter Controls */}
      {showFilters && (
        <div className="filter-controls">
        <div className="filter-section">
          <label htmlFor="instrument-filter">Instrument:</label>
          <select 
            id="instrument-filter"
            value={selectedInstrument} 
            onChange={(e) => setSelectedInstrument(e.target.value)}
            className="instrument-dropdown"
          >
            <option value="all">All Instruments</option>
            {availableInstruments.map(instrument => (
              <option key={instrument} value={instrument}>
                {instrument.charAt(0).toUpperCase() + instrument.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-section">
          <label htmlFor="duration-slider">
            Duration: {Math.floor(durationRange[0] / 60)}:{String(durationRange[0] % 60).padStart(2, '0')} - {Math.floor(durationRange[1] / 60)}:{String(durationRange[1] % 60).padStart(2, '0')}
          </label>
          <div className="dual-slider">
            <input
              type="range"
              id="duration-slider"
              min="0"
              max={maxDuration}
              value={durationRange[0]}
              onChange={(e) => setDurationRange([Math.min(parseInt(e.target.value), durationRange[1] - 10), durationRange[1]])}
              className="slider slider-min"
            />
            <input
              type="range"
              min="0"
              max={maxDuration}
              value={durationRange[1]}
              onChange={(e) => setDurationRange([durationRange[0], Math.max(parseInt(e.target.value), durationRange[0] + 10)])}
              className="slider slider-max"
            />
          </div>
        </div>

        <div className="filter-section">
          <label htmlFor="parts-slider">
            Parts: {partCountRange[0]} - {partCountRange[1]}
          </label>
          <div className="dual-slider">
            <input
              type="range"
              id="parts-slider"
              min="1"
              max={maxParts}
              value={partCountRange[0]}
              onChange={(e) => setPartCountRange([Math.min(parseInt(e.target.value), partCountRange[1] - 1), partCountRange[1]])}
              className="slider slider-min"
            />
            <input
              type="range"
              min="1"
              max={maxParts}
              value={partCountRange[1]}
              onChange={(e) => setPartCountRange([partCountRange[0], Math.max(parseInt(e.target.value), partCountRange[0] + 1)])}
              className="slider slider-max"
            />
          </div>
        </div>

        <div className="filter-section">
          <label htmlFor="pages-slider">
            Pages: {pageCountRange[0]} - {pageCountRange[1]}
          </label>
          <div className="dual-slider">
            <input
              type="range"
              id="pages-slider"
              min="1"
              max={maxPages}
              value={pageCountRange[0]}
              onChange={(e) => setPageCountRange([Math.min(parseInt(e.target.value), pageCountRange[1] - 1), pageCountRange[1]])}
              className="slider slider-min"
            />
            <input
              type="range"
              min="1"
              max={maxPages}
              value={pageCountRange[1]}
              onChange={(e) => setPageCountRange([pageCountRange[0], Math.max(parseInt(e.target.value), pageCountRange[0] + 1)])}
              className="slider slider-max"
            />
          </div>
        </div>
        </div>
      )}
      
      <SongList
        songs={currentResults}
        handleSongClick={handleSongClick}
        toggleFavorite={toggleFavorite}
        isFavorite={isFavorite}
      />
      
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
