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
  const [searchResults, setSearchResults] = useState([]);
  const [favorites, setFavorites] = useState(() => {
    try {
      const raw = localStorage.getItem('chordquest.favorites');
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  });

  // Persist favorites to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('chordquest.favorites', JSON.stringify(favorites));
    } catch (e) {
      // ignore localStorage write errors (e.g., storage full or disabled)
    }
  }, [favorites]);
  const [currentSongId, setCurrentSongId] = useState(null);

  // Placeholder data for music sheets
  const placeholderSongs = [
    { id: 1, title: "Bohemian Rhapsody", artist: "Queen", difficulty: "Advanced", key: "Bb Major" },
    { id: 2, title: "Imagine", artist: "John Lennon", difficulty: "Intermediate", key: "C Major" },
    { id: 3, title: "Hotel California", artist: "Eagles", difficulty: "Intermediate", key: "Bm" },
    { id: 4, title: "Let It Be", artist: "The Beatles", difficulty: "Beginner", key: "C Major" },
    { id: 5, title: "Wonderwall", artist: "Oasis", difficulty: "Beginner", key: "Em" },
  ];

  // Recommended songs (placeholder)
  const recommendedSongs = [
    { id: 6, title: "Sweet Child O' Mine", artist: "Guns N' Roses", difficulty: "Intermediate", key: "D Major" },
    { id: 7, title: "Stairway to Heaven", artist: "Led Zeppelin", difficulty: "Advanced", key: "Am" },
    { id: 8, title: "Hey Jude", artist: "The Beatles", difficulty: "Beginner", key: "F Major" },
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
    const allSongs = [...searchResults, ...placeholderSongs, ...recommendedSongs];
    return allSongs.find(song => song.id === currentSongId);
  };

  const getSimilarSongs = () => {
    const allSongs = [...searchResults, ...placeholderSongs, ...recommendedSongs];
    const otherSongs = allSongs.filter(song => song.id !== currentSongId);
    return otherSongs.slice(0, 3);
  };

  const toggleFavorite = (e, songId) => {
    e.stopPropagation(); // Prevent song click when clicking favorite button
    setFavorites(prev => (
      prev.includes(songId) ? prev.filter(id => id !== songId) : [...prev, songId]
    ));
  };

  const isFavorite = (songId) => favorites.includes(songId);

  const getFavoriteSongs = () => {
    const allSongs = [...searchResults, ...placeholderSongs, ...recommendedSongs];
    return allSongs.filter(song => favorites.includes(song.id));
  };

  return (
    <div className="App">
      <Navbar setCurrentView={setCurrentView} />

      <main className="main-content">
        {currentView === 'home' && (
          <HomeView
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            handleSearch={handleSearch}
            recommendedSongs={recommendedSongs}
            handleSongClick={handleSongClick}
            toggleFavorite={toggleFavorite}
            isFavorite={isFavorite}
          />
        )}

        {currentView === 'results' && (
          <ResultsView
            searchResults={searchResults}
            setCurrentView={setCurrentView}
            handleSongClick={handleSongClick}
            toggleFavorite={toggleFavorite}
            isFavorite={isFavorite}
          />
        )}

        {currentView === 'favorites' && (
          <FavoritesView
            favoriteSongs={getFavoriteSongs()}
            setCurrentView={setCurrentView}
            handleSongClick={handleSongClick}
            toggleFavorite={toggleFavorite}
            isFavorite={isFavorite}
          />
        )}

        {currentView === 'songDetail' && (
          <SongDetailView
            song={getCurrentSong()}
            setCurrentView={setCurrentView}
            previousView={previousView}
            toggleFavorite={toggleFavorite}
            isFavorite={isFavorite}
            handleSongClick={handleSongClick}
            recommendedSongs={getSimilarSongs()}
          />
        )}
      </main>
    </div>
  );
}

export default App;
