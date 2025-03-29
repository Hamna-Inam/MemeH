#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import open_clip
import os
import numpy as np
import faiss
import json
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import easyocr
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, UploadFile, File


# In[3]:


app = FastAPI()


# In[4]:


#device = "cuda" if torch.cuda.is_available() else "cpu"
#model_tuple = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
#model = model_tuple[0]
#preprocess = model_tuple[1]
#model.to(device)

model = SentenceTransformer('clip-ViT-B-32')  # CLIP used to convert text (captions) into dense vectors.


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # CLIP's vision and text model from Hugging Face
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") # Handles resizing, normalizing, and tokenizing images/text before feeding them into the model


# In[5]:


reader = easyocr.Reader(['en'])

def extract_text_easyocr(image_path):
    """Extract text from an image using EasyOCR."""
    text_list = reader.readtext(image_path, detail=0)  # Extract text
    return " ".join(text_list) if text_list else "No text found"  # Concatenate text


# In[12]:


meme_folder = "memes/"
meme_database = {}
filenames = []
dimension = 512
index = faiss.IndexFlatIP(dimension)  # Inner Product index


# Normalize embeddings before adding them to FAISS (needed for cosine similarity)
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def store_memes():
    global filenames, index 
    
    for file in os.listdir(meme_folder):
        if file.endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(meme_folder, file)
            
            # 1️⃣ **Extract Image Embeddings** 
            image = Image.open(file_path).convert("RGB")  # Load image
            inputs = clip_processor(images=image, return_tensors="pt")  # Preprocess image and convert it to a tensor 
            with torch.no_grad(): #Disables gradient calculations to speed up inference
                image_embedding = clip_model.get_image_features(**inputs)  # Passes the preprocessed image into CLIP’s vision model and get an image vector
            image_embedding = image_embedding.squeeze(0).numpy()  # Removes batch dimension from the vector and Converts it into a numpy array

            # 2️⃣ **Extract Text Embeddings**  
            caption = extract_text_easyocr(file_path)  # OCR for caption
            text_embedding = model.encode(caption)  # Convert text to embedding

            # 3️⃣ **Store Embeddings in Database**  
            meme_database[file] = {
                "image_embedding": image_embedding,
                "text_embedding": text_embedding
            }
            filenames.append(file)

    print("✅ Stored", len(meme_database), "memes as dense vectors!")

    # Convert embeddings into FAISS format (Normalize for cosine similarity)
    vectors = np.array([data["image_embedding"] for data in meme_database.values()], dtype="float32")
    vectors = normalize(vectors)  # Normalize for cosine similarity

    # FAISS index for cosine similarity (Inner Product)
    index.add(vectors)  # Add normalized vectors

    print("✅ FAISS index (Cosine Similarity) created with", len(meme_database), "memes!")


# In[13]:


store_memes()


# In[14]:


def show_meme(meme):
    image_path = os.path.join(meme_folder, meme)  # Get full path to meme image
    image = Image.open(image_path)  # Open the image
    
    # Display the image with caption
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()


# In[15]:


# Function to search meme using cosine similarity
@app.get("/search")
def search_meme(query):
    query_embedding = model.encode(query).astype("float32")  # Convert query to vector
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize

    # FAISS search (dot product ~ cosine similarity)
    _, indices = index.search(np.array([query_embedding]), k=1)  # Find closest meme

    best_match = filenames[indices[0][0]]  # Get filename of best meme
    return best_match


# In[19]:


@app.post("/upload/")
async def upload_meme(file: UploadFile = File(...)):
    """Upload a new meme and add it to the search index."""
    filepath = os.path.join(meme_folder, file.filename)
    
    with open(filepath, "wb") as f:
        f.write(await file.read())

    caption = extract_text_easyocr(filepath)
    embedding = model.encode(caption)
    
    meme_database[file.filename] = embedding
    filenames.append(file.filename)
    index.add(np.array([embedding]).astype("float32"))
    
    return {"message": f"{file.filename} added successfully!"}




# In[31]:


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dense:app", host="127.0.0.1", port=8000, reload=True)


# In[199]:


user_prompt = "being the eldest daughter"
best_meme = search_meme(user_prompt)
show_meme(best_meme)  

