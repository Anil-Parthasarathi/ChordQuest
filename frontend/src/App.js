import React, { useState, useEffect } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import HomeView from './components/HomeView';
import ResultsView from './components/ResultsView';
import FavoritesView from './components/FavoritesView';
import SongDetailView from './components/SongDetailView';

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [currentView, setCurrentView] = useState('home'); // 'home', 'results', 'favorites', 'songDetail'
  const [previousView, setPreviousView] = useState('home'); // Track the view to return to from song detail
  const [currentPage, setCurrentPage] = useState(1); // Track current page for results
  const [searchResults, setSearchResults] = useState([]);
  const [favoriteSongs, setFavoriteSongs] = useState([]);

  // Load favorites from backend on startup
  useEffect(() => {
    const loadFavorites = async () => {
      try {
        const res = await fetch('/api/retrieveFavorites');
        if (res.ok) {
          const data = await res.json();
          setFavoriteSongs(data.favorites || []);
        }
      } catch (err) {
        console.error('Failed to load favorites:', err);
      }
    };
    loadFavorites();
  }, []);

  const [currentSongId, setCurrentSongId] = useState(null);

  // Placeholder data for music sheets
  const placeholderSongs = [
    { id: 1, title: "Bohemian Rhapsody", artist: "Queen", difficulty: "Advanced", key: "Bb Major" },
    { id: 2, title: "Imagine", artist: "John Lennon", difficulty: "Intermediate", key: "C Major" },
    { id: 3, title: "Hotel California", artist: "Eagles", difficulty: "Intermediate", key: "Bm" },
    { id: 4, title: "Let It Be", artist: "The Beatles", difficulty: "Beginner", key: "C Major" },
    { id: 5, title: "Wonderwall", artist: "Oasis", difficulty: "Beginner", key: "Em" },
  ];

  // Search function
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      return;
    }

    try {
      const res = await fetch(`/api/search?query=${encodeURIComponent(searchQuery)}`);
      if (res.ok) {
        const data = await res.json();
        console.log('Search results from backend:', data.result);
        setSearchResults(data.result);
        setCurrentPage(1);
        setCurrentView('results');
      } else {
        console.error('Backend search returned error status:', res.status);
        setSearchResults([]);
        setCurrentView('results');
      }
    } catch (err) {
      console.error('Backend search failed:', err);
      setSearchResults([]);
      setCurrentView('results');
    }
  };

  const handleSongClick = (songId) => {
    // Save current view before navigating to song detail
    if (currentView !== 'songDetail') {
      setPreviousView(currentView);
    }
    setCurrentSongId(songId);
    setCurrentView('songDetail');
  };

  const getCurrentSong = () => {
    const allSongs = [...searchResults, ...placeholderSongs, ...favoriteSongs];
    return allSongs.find(song => song.id === currentSongId);
  };

  const getSimilarSongs = () => {
    const allSongs = [...searchResults, ...placeholderSongs, ...favoriteSongs];
    const otherSongs = allSongs.filter(song => song.id !== currentSongId);
    return otherSongs.slice(0, 3);
  };

  const toggleFavorite = async (e, songId) => {
    e.stopPropagation(); // Prevent song click when clicking favorite button
    
    const isCurrentlyFavorite = isFavorited(songId);
    
    try {
      if (isCurrentlyFavorite) {
        // Remove from favorites
        const res = await fetch('/api/removeFavorite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: songId })
        });
        
        if (res.ok) {
          const data = await res.json();
          setFavoriteSongs(data.favorites || []);
        }
      } else {
        // Add to favorites - need to get full song data
        const allSongs = [...searchResults, ...placeholderSongs, ...favoriteSongs];
        const song = allSongs.find(s => s.id === songId);
        
        if (song) {
          const res = await fetch('/api/setfavorite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(song)
          });
          
          if (res.ok) {
            const data = await res.json();
            setFavoriteSongs(data.favorites || []);
          }
        }
      }
    } catch (err) {
      console.error('Failed to toggle favorite:', err);
    }
  };

  const isFavorited = (songId) => favoriteSongs.some(song => song.id === songId);

  return (
    <div className="App">
      <Navbar setCurrentView={setCurrentView} />

      <main className="main-content">
        {currentView === 'home' && (
          <HomeView
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            handleSearch={handleSearch}
            recommendedSongs={favoriteSongs}
            handleSongClick={handleSongClick}
            toggleFavorite={toggleFavorite}
            isFavorite={isFavorited}
          />
        )}

      {currentView === 'results' && (
        <ResultsView 
          searchResults={searchResults}
          setCurrentView={setCurrentView}
          handleSongClick={handleSongClick}
          toggleFavorite={toggleFavorite}
          isFavorite={isFavorited}
          currentPage={currentPage}
          setCurrentPage={setCurrentPage}
        />
      )}        {currentView === 'favorites' && (
          <FavoritesView
            favoriteSongs={favoriteSongs}
            setCurrentView={setCurrentView}
            handleSongClick={handleSongClick}
            toggleFavorite={toggleFavorite}
            isFavorite={isFavorited}
          />
        )}

        {currentView === 'songDetail' && (
          <SongDetailView
            song={getCurrentSong()}
            setCurrentView={setCurrentView}
            previousView={previousView}
            toggleFavorite={toggleFavorite}
            isFavorite={isFavorited}
            handleSongClick={handleSongClick}
            recommendedSongs={getSimilarSongs()}
          />
        )}
      </main>
    </div>
  );
}

export default App;
