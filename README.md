# ChordQuest

The new go-to source for finding music sheets!

ChordQuest is search engine / recommender system that will help you discover your next rocking performance!!

## Project Structure

```
ChordQuest/
├── backend/           # Flask backend
│   ├── app.py        # Main Flask application
│   ├── requirements.txt
│   └── .gitignore
├── frontend/         # React frontend
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── .gitignore
└── README.md
```

## Setup Instructions

### Backend Setup (Flask)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Flask server:
   ```bash
   python app.py
   ```

   The backend will start on `http://localhost:5000`

### Frontend Setup (React)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The frontend will start on `http://localhost:3000`

## Running the Application

1. Start the Flask backend (in one terminal):
   ```bash
   cd backend
   source venv/bin/activate
   python app.py
   ```

2. Start the React frontend (in another terminal):
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser to `http://localhost:3000` to see the app running

## API Endpoints

- `GET /api/health` - Health check endpoint
- `GET /api/hello` - Sample hello world endpoint

## Tech Stack

- **Frontend**: React.js 18.2
- **Backend**: Python Flask 3.0
- **CORS**: flask-cors for cross-origin requests
