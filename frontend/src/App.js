import React, { useState } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import HomeView from './components/HomeView';
import ResultsView from './components/ResultsView';
import FavoritesView from './components/FavoritesView';
import SongDetailView from './components/SongDetailView';

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [currentView, setCurrentView] = useState('home'); // 'home', 'results', 'favorites', 'songDetail'
  const [searchResults, setSearchResults] = useState([]);
  const [favorites, setFavorites] = useState([]);
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

  // Placeholder search function, just simple binary search for now
  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Filter placeholder results based on search query
      const filtered = placeholderSongs.filter(song =>
        song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        song.artist.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setSearchResults(filtered);
      setCurrentView('results');
    }
  };

  const handleSongClick = (songId) => {
    setCurrentSongId(songId);
    setCurrentView('songDetail');
  };

  const getCurrentSong = () => {
    const allSongs = [...placeholderSongs, ...recommendedSongs];
    return allSongs.find(song => song.id === currentSongId);
  };

  const getSimilarSongs = () => {
    // Get 3 random songs as similar songs (placeholder logic)
    const allSongs = [...placeholderSongs, ...recommendedSongs];
    const otherSongs = allSongs.filter(song => song.id !== currentSongId);
    return otherSongs.slice(0, 3);
  };

  const toggleFavorite = (e, songId) => {
    e.stopPropagation(); // Prevent song click when clicking favorite button
    if (favorites.includes(songId)) {
      setFavorites(favorites.filter(id => id !== songId));
    } else {
      setFavorites([...favorites, songId]);
    }
  };

  const isFavorite = (songId) => favorites.includes(songId);

  const getFavoriteSongs = () => {
    return [...placeholderSongs, ...recommendedSongs].filter(song => favorites.includes(song.id));
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
