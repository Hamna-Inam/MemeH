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

# %%
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model_tuple = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
#model = model_tuple[0]
#preprocess = model_tuple[1]
#model.to(device)

model = SentenceTransformer('clip-ViT-B-32')  # CLIP used to convert text (captions) into dense vectors.

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # CLIP's vision and text model from Hugging Face
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast='True') # Handles resizing, normalizing, and tokenizing images/text before feeding them into the model

meme_folder = "memes/"  # meme folder
FAISS_INDEX_PATH = "faiss.index.bin" #store FAISS indices
MEME_DB_PATH = "meme_database.pkl"

#meme_database = {} #metadata and embeddings for each meme 
filenames = [] #to map FAISS results back to real filenames
index = None  # FAISS index

# Mount static files
os.makedirs(meme_folder, exist_ok=True)
app.mount("/memes", StaticFiles(directory=meme_folder), name="memes")

def save_index():
    faiss.write_index(index, FAISS_INDEX_PATH) # save faiss index
    with open( MEME_DB_PATH, "wb") as f:
        pickle.dump({"filenames":filenames},f) # save memes and their corresponding filenames
    print("FAISS index and meme database saved!")

def load_index():
    # Load FAISS index and meme database from disk
    global index, filenames
    
    if os.path.exists(FAISS_INDEX_PATH) :
        print("Loading existing FAISS index and meme database")
        index = faiss.read_index(FAISS_INDEX_PATH) #Load precomputed FAISS indices
        print("FAISS index loaded with", len(filenames), "memes!")
        return True
    return False

# Normalize embeddings before adding them to FAISS (needed for cosine similarity)
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def store_memes():
    global filenames, index
    vectors = [] 
    
    for file in os.listdir(meme_folder):
        if file.endswith((".jpg", ".png", ".jpeg",".JPG",".PNG",".JPEG")):
            file_path = os.path.join(meme_folder, file)
            
            # **Extract Image Embeddings** 
            image = Image.open(file_path).convert("RGB")  # Load image
            inputs = clip_processor(images=image, return_tensors="pt")  # Preprocess image and convert it to a tensor 
            with torch.no_grad(): #Disables gradient calculations to speed up inference
                image_embedding = clip_model.get_image_features(**inputs)  # Passes the preprocessed image into CLIP's vision model and get an image vector
            image_embedding = image_embedding.squeeze(0).numpy()  # Removes batch dimension from the vector and Converts it into a numpy array

            # **Store Embeddings in Database**  
            #meme_database[file] = {  # key = filename, value = image embedding
            #    "image_embedding": image_embedding,}
            filenames.append(file)
            vectors.append(image_embedding)

    print("Stored", len(filenames), "memes as dense vectors!")

    # Convert embeddings into FAISS format (Normalize for cosine similarity)
    vectors = np.array(vectors, dtype="float32")
    vectors = normalize(vectors)  # Normalize for cosine similarity

    # FAISS index for cosine similarity (Inner Product)
    index.add(vectors)  # Add normalized vectors

    print("FAISS index (Cosine Similarity) created with", len(vectors), "memes!")

    save_index()

@app.get("/search")
def search_meme(query: str, threshold: float = 0.5, k: int = 1):
    query_embedding = model.encode(query).astype("float32")  # Convert the search text to a numerical embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize the embedding

    # FAISS search (returns k nearest memes)
    similarities, indices = index.search(np.array([query_embedding]), k=k)
    
    # Extract matches and their scores
    # Map each FAISS index back to the corresponding meme file.

    results = [
        {"meme": filenames[idx], "similarity": float(sim)}
        for idx, sim in zip(indices[0], similarities[0])
        if sim >= 0  # Avoid invalid indices
    ]


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
    
    # Save the file
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Extract image embedding
    image = Image.open(filepath).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    
    embedding = embedding.squeeze(0).numpy()  # Convert to numpy

    # Normalize embedding (important for FAISS similarity search)
    embedding /= np.linalg.norm(embedding)
    
     # Update database and FAISS index
    # meme_database[unique_filename] = {
    #     "image_embedding": embedding,
    #    }
    filenames.append(unique_filename)
    index.add(np.array([embedding]).astype("float32"))

    #np.save("filenames.npy", np.array(filenames))  # Save updated filenames
    faiss.write_index(index, FAISS_INDEX_PATH)
    #with open(MEME_DB_PATH, "wb") as f:
    #    pickle.dump(meme_database, f)
    
    return {"message": f"{file.filename} added successfully!"}

def index_new_memes():
    """Index only new memes that aren't already in the database"""
    global filenames, index
    #meme_database
    
    # Get list of all memes in the folder
    current_memes = set(os.listdir(meme_folder))
    # Get list of already indexed memes
    indexed_memes = set(filenames)
    # Find new memes that aren't indexed yet
    new_memes = current_memes - indexed_memes
    
    if not new_memes:
        print("No new memes to index!")
        return
    
    print(f"Found {len(new_memes)} new memes to index...")
    
    for file in new_memes:
        if file.endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(meme_folder, file)
            
            # Extract Image Embeddings
            image = Image.open(file_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_embedding = clip_model.get_image_features(**inputs)
            image_embedding = image_embedding.squeeze(0).numpy()
            
            # Normalize embedding
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            
            # Store in database
            #meme_database[file] = {
            #    "image_embedding": image_embedding,
            #}
            filenames.append(file)
            
            # Add to FAISS index
            index.add(np.array([image_embedding]).astype("float32"))
            
            print(f"Indexed: {file}")
    
    print(f"Successfully indexed {len(new_memes)} new memes!")
    save_index()



index = faiss.IndexFlatIP(512)
store_memes()
save_index()

'''
#if not load_index():
    print("No existing index found. Creating new index")
    store_memes()
    save_index()
else:
    print("Successfully loaded existing index with memes!")
'''