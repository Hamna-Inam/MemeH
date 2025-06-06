
**Skibidi Rizz Search Engine** is a lightweight AI-powered meme retrieval tool. It uses CLIP and FAISS to let users search for memes using natural language. Want a meme about “procrastination”? Just type it in — and it’ll fetch the closest match!

## Features

* Search memes by description (e.g., *"when you love coffee"*)
* Upload your own memes
* Automatically indexes new memes
* FAISS for fast similarity search
* Works locally, with a small sample meme set

---

## Usage

### Search for Memes

Send a GET request to `/search?query=your text`:

```bash
curl "http://127.0.0.1:8000/search?query=when you hate coffee"
```

### Upload a Meme

Use the `/upload/` endpoint with an image file.

---

## Tech Stack

* FastAPI
* SentenceTransformers (`clip-ViT-L-14`)
* FAISS

---

## Notes

* First run creates the index from memes in `/memes/`
* After that, only new memes are indexed on startup
* You can drop in new images anytime — just refresh the app

---


## Author

Made by Hamna Inam
Inspired by a bad DAA MidTerm
