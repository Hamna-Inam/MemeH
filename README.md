# Skibidi Rizz Search Engine

A lightweight AI-powered meme retrieval tool. It uses CLIP and FAISS to let users search for memes using natural language. Want a meme about "procrastination"? Just type it in — and it'll fetch the closest match!

## Features
- Search memes by description (e.g., "when you love coffee")
- Upload your own memes
- Automatically indexes new memes
- FAISS for fast similarity search
- Works locally, with a small sample meme set

## Installation

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Usage

### Search for Memes
Send a GET request to `/search?query=your text`:
```bash
curl "http://127.0.0.1:8000/search?query=when you hate coffee"
```

### Upload a Meme
Use the `/upload/` endpoint with an image file.

## Project Structure
```
MemeH/
├── README.md
├── requirements.txt
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── meme_database.pkl      # Stored meme data
│   ├── faiss.index.bin        # FAISS index for similarity search
│   ├── memes/                 # Meme storage directory
│   └── old_memes/             # Archive of old memes
├── frontend/                  # React frontend
│   ├── build/                 # Production build files
│   ├── node_modules/          # Node.js dependencies
│   ├── public/                # Static assets
│   ├── src/                   # React source code
│   ├── package.json           # Frontend dependencies
│   ├── package-lock.json
│   └── .gitignore
└── .gitignore
```

## Tech Stack
- Flask (you mentioned FastAPI but your file is app.py which suggests Flask)
- SentenceTransformers (`clip-ViT-L-14`)
- FAISS
- React (frontend)

## Notes
- First run creates the index from memes in `/memes/`
- After that, only new memes are indexed on startup
- You can drop in new images anytime — just refresh the app

## Author
Made by Hamna Inam  
Inspired by a bad DAA MidTerm