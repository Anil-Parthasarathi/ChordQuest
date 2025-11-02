import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState('Loading...');

  useEffect(() => {
    // Test connection to Flask backend
    fetch('/api/health')
      .then(response => response.json())
      .then(data => {
        setStatus(data.message);
      })
      .catch(error => {
        setStatus('Failed to connect to backend');
        console.error('Error:', error);
      });

    // Fetch sample message
    fetch('/api/hello')
      .then(response => response.json())
      .then(data => {
        setMessage(data.message);
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>ChordQuest</h1>
        <p>The future of musical performance is here!</p>
        <div className="status-container">
          <p><strong>Backend Status:</strong> {status}</p>
          <p><strong>API Message:</strong> {message}</p>
        </div>
      </header>
    </div>
  );
}

export default App;
