import torch
import os
import numpy as np
import faiss
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, UploadFile, File
import pickle
import uuid
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all frontend requests 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clip_model = SentenceTransformer('clip-ViT-L-14')  # CLIP used to convert search query text into dense vectors.

meme_folder = "memes/"  # meme folder
FAISS_INDEX_PATH = "faiss.index.bin" #store FAISS indices
MEME_DB_PATH = "meme_database.pkl"

filenames = [] #to map FAISS results back to real filenames
index = None  # FAISS index

# Mount static files
os.makedirs(meme_folder, exist_ok=True)
app.mount("/memes", StaticFiles(directory=meme_folder), name="memes")

def save_index():
    faiss.write_index(index, FAISS_INDEX_PATH) # save faiss index
    with open(MEME_DB_PATH, "wb") as f:
        pickle.dump({"filenames": filenames}, f) # save memes and their corresponding filenames
    print("FAISS index and meme database saved!")

def load_index():
    # Load FAISS index and meme database from disk
    global index, filenames
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(MEME_DB_PATH):
        print("Loading existing FAISS index and meme database")
        index = faiss.read_index(FAISS_INDEX_PATH) #Load precomputed FAISS indices
        
        # Load the filenames from pickle file
        with open(MEME_DB_PATH, "rb") as f:
            data = pickle.load(f)
            filenames.extend(data["filenames"])
        
        print("FAISS index loaded with", len(filenames), "memes!")
        return True
    return False

def store_memes():
    global filenames, index
    vectors = [] 
    
    for file in os.listdir(meme_folder):
        if file.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")):
            file_path = os.path.join(meme_folder, file)
            
            # Extract Image Embeddings
            image = Image.open(file_path).convert("RGB")  # Load image
            image_embedding = clip_model.encode(image, convert_to_numpy=True, normalize=True) # Passes the image into CLIP's vision model and get an image vector
            filenames.append(file)  # Use the actual filename from disk
            vectors.append(image_embedding)

    print("Stored", len(filenames), "memes as dense vectors!")

    if vectors:  # Only proceed if we have vectors
        # Convert embeddings into FAISS format
        vectors = np.array(vectors, dtype="float32")

        # FAISS index for cosine similarity (Inner Product)
        index.add(vectors)  # Add normalized vectors

        print("FAISS index (Cosine Similarity) created with", len(vectors), "memes!")

@app.get("/search")
def search_meme(query: str, threshold: float = 0.5, k: int = 5):  # Increased k for debugging
    if index is None or len(filenames) == 0:
        return {"error": "No memes indexed yet. Please upload some memes first."}

    # encode and normalize the query
    query_embedding = clip_model.encode(query, convert_to_numpy=True, normalize=True).astype("float32") 

    # FAISS search (return k nearest memes)
    similarities, indices = index.search(np.array([query_embedding]), k=k)
    
    # Debug: Print search results
    print(f"Search query: {query}")
    print(f"Found indices: {indices[0]}")
    print(f"Similarities: {similarities[0]}")
    print(f"Total filenames in DB: {len(filenames)}")
    
    # Extract matches and their scores
    # Map each FAISS index back to the corresponding meme file.
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx < len(filenames) and sim >= 0:  # Avoid invalid indices
            filename = filenames[idx]
            print(f"Mapping index {idx} to filename: {filename}")
            
            # Verify file exists on disk
            file_path = os.path.join(meme_folder, filename)
            if os.path.exists(file_path):
                results.append({"meme": filename, "similarity": float(sim)})
                print(f"File exists: {filename}")
            else:
                print(f"File missing: {filename} at {file_path}")

    # If the best match meets the threshold, return it
    if results and results[0]["similarity"] >= threshold:
        return {"best_match": results[0], "similar_memes": results[1:]}

    # If no match meets the threshold, return similar memes + upload suggestion
    return {
        "message": "No strong match found. Here are some similar memes:",
        "similar_memes": results,
        "upload_suggestion": "Can't find what you're looking for? Upload a meme!"
    }

@app.post("/upload/")
async def upload_meme(file: UploadFile = File(...)):
    """Upload a new meme and add it to the search index."""

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(meme_folder, unique_filename)
    
    print(f"Uploading file: {file.filename}")
    print(f"Saving as: {unique_filename}")
    
    # Save the file
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Verify file was saved
    if os.path.exists(filepath):
        print(f"File saved successfully: {filepath}")
    else:
        print(f"Failed to save file: {filepath}")
        return {"error": "Failed to save file"}

    # Extract image embedding
    image = Image.open(filepath).convert("RGB")
    
    embedding = clip_model.encode(image, convert_to_numpy=True, normalize=True) # Passes the image into CLIP's vision model and get an image vector

    filenames.append(unique_filename)
    index.add(np.array([embedding]).astype("float32"))

    save_index()  # Save both FAISS index and filenames
    
    print(f"Added to database: {unique_filename}")
    return {"message": f"{file.filename} uploaded and indexed successfully as {unique_filename}!"}


if not load_index():
    print("No existing index found. Creating new index")
    store_memes()
    save_index()

# If you add memes to meme folder and want to use those, call store_memes() and save_index() without load_index() just once
# so comment out the part above and uncomment the lines below (just once)
# store_memes()
# save_index()