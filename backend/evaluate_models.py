import os
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
import pickle

MEME_FOLDER = "memes/"
FAISS_INDEX_PATH = "faiss.index.bin"
MEME_DB_PATH = "meme_database.pkl"
GROUND_TRUTH_FILE = "ground_truth.txt"

# Models to evaluate
MODELS_TO_EVALUATE = [
    'clip-ViT-B-32',
    'clip-ViT-L-14',
    'clip-ViT-B-16', 
]


def load_ground_truth(file_path):
    """Loads queries and their expected relevant meme from a file."""
    ground_truth_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            query, relevant_meme = line.strip().split(',')
            ground_truth_data[query] = relevant_meme
    return ground_truth_data

def build_index_for_model(model_name):
    """
    Builds a new FAISS index and meme database for a given model.
    This will overwrite existing index files.
    """
    print(f"\n--- Building index for model: {model_name} ---")
    current_clip_model = SentenceTransformer(model_name)

    # Get embedding dimension dynamically for the current model
    sample_embedding = current_clip_model.encode("test", convert_to_numpy=True, normalize=True)
    embedding_dim = sample_embedding.shape[0]

    global filenames, index # Refer to global variables for consistency with your app
    filenames = []
    index = faiss.IndexFlatIP(embedding_dim)

    vectors = []
    for file in os.listdir(MEME_FOLDER):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            file_path = os.path.join(MEME_FOLDER, file)
            try:
                image = Image.open(file_path).convert("RGB")
                image_embedding = current_clip_model.encode(image, convert_to_numpy=True, normalize=True)
                filenames.append(file)
                vectors.append(image_embedding)
            except Exception as e:
                print(f"Could not process image {file_path}: {e}")
                continue

    if vectors:
        vectors = np.array(vectors, dtype="float32")
        index.add(vectors)
        print(f"Indexed {len(filenames)} memes with {model_name}.")

        # Save the new index and filenames specific to this model for potential debugging

        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(MEME_DB_PATH, "wb") as f:
            pickle.dump({"filenames": filenames}, f)
        print(f"Index for {model_name} saved.")
    else:
        print(f"No memes found or processed for indexing with {model_name}.")
        filenames = [] # Ensure filenames is empty if no memes were processed

    return current_clip_model, index, filenames


def search_meme_for_evaluation(query, current_clip_model, current_index, current_filenames, k=1):
    """
    Performs a search using the provided model and index,
    similar to your FastAPI search endpoint.
    """
    if current_index is None or len(current_filenames) == 0:
        return []

    try:
        query_embedding = current_clip_model.encode(query, convert_to_numpy=True, normalize=True).astype("float32")
    except Exception as e:
        print(f"Error encoding query '{query}' with current model: {e}")
        return []

    similarities, indices = current_index.search(np.array([query_embedding]), k=k)

    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        if idx < len(current_filenames): # Ensure index is valid
            results.append({"meme": current_filenames[idx], "similarity": float(sim)})
    return results

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    if not ground_truth:
        print(f"Error: No ground truth data found in {GROUND_TRUTH_FILE}. Please create it.")
        exit()

    print(f"Loaded {len(ground_truth)} queries from ground truth.")

    for model_name in MODELS_TO_EVALUATE:
        # Build index for the current model
        active_clip_model, active_index, active_filenames = build_index_for_model(model_name)

        if not active_filenames:
            print(f"Skipping evaluation for {model_name} as no memes were indexed.")
            continue

        correct_predictions = 0
        total_queries = len(ground_truth)

        for query, expected_meme in ground_truth.items():
            search_results = search_meme_for_evaluation(query, active_clip_model, active_index, active_filenames, k=1)

            if search_results:
                best_match_meme = search_results[0]["meme"]
                if best_match_meme == expected_meme:
                    correct_predictions += 1
                    # print(f"Query: '{query}' - Correct! Matched '{best_match_meme}'")
                # else:
                    # print(f"Query: '{query}' - Incorrect. Expected '{expected_meme}', got '{best_match_meme}'")
            # else:
                # print(f"Query: '{query}' - No results found.")

        # Calculate Precision@1
        precision_at_1 = (correct_predictions / total_queries) if total_queries > 0 else 0

        print(f"\n--- Evaluation Results for {model_name} ---")
        print(f"Total queries: {total_queries}")
        print(f"Correct P@1 predictions: {correct_predictions}")
        print(f"Precision@1 (P@1): {precision_at_1:.4f}")
        print("-" * 40)

    print("\nEvaluation complete!")