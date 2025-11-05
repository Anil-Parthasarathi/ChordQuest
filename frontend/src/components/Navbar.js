import React from 'react';

function Navbar({ setCurrentView }) {
  return (
    <nav className="navbar">
      <div className="navbar-content">
        <div className="navbar-brand" onClick={() => setCurrentView('home')}>
          <img
            src={process.env.PUBLIC_URL + '/assets/chordQuestFavicon.png'}
            alt="ChordQuest Logo"
            className="navbar-logo"
          />
          <h1>ChordQuest</h1>
        </div>
        <button 
          className="favorites-btn"
          onClick={() => setCurrentView('favorites')}
        >
          ★ Favorites
        </button>
      </div>
    </nav>
  );
}

export default Navbar;
