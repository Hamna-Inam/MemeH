import { useState } from "react";

export default function MemeApp() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [file, setFile] = useState(null);

  // Handle text search
  const handleSearch = async () => {
    const res = await fetch(`http://localhost:8000/search?query=${query}`);
    const data = await res.json();
    setResults(data);
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    await fetch("http://localhost:8000/upload/", {
      method: "POST",
      body: formData,
    });
    alert("Meme uploaded successfully!");
  };

  // Inline styles to ensure the styles are applied correctly
  const styles = {
    container: {
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
      backgroundColor: "#fce7f3", // Light pink background
      padding: "20px"
    },
    card: {
      padding: "32px",
      backgroundColor: "white",
      boxShadow: "0 10px 25px rgba(0, 0, 0, 0.1)",
      borderRadius: "12px",
      width: "100%",
      maxWidth: "500px",
      textAlign: "center"
    },
    heading: {
      fontSize: "28px",
      fontWeight: "bold",
      marginBottom: "24px",
      color: "#374151"
    },
    searchInput: {
      width: "100%",
      padding: "16px",
      borderRadius: "9999px",
      border: "1px solid #d1d5db",
      marginBottom: "16px"
    },
    searchButton: {
      backgroundColor: "#ec4899",
      color: "white",
      padding: "12px",
      width: "100%",
      marginTop: "16px",
      borderRadius: "8px",
      border: "none",
      cursor: "pointer"
    },
    uploadSection: {
      marginTop: "24px",
      padding: "16px",
      backgroundColor: "#f9fafb",
      borderRadius: "8px"
    },
    uploadTitle: {
      fontSize: "18px",
      fontWeight: "500",
      marginBottom: "12px",
      color: "#4b5563"
    },
    fileInput: {
      width: "100%",
      padding: "12px",
      border: "1px solid #d1d5db",
      borderRadius: "8px"
    },
    uploadButton: {
      backgroundColor: "#ec4899",
      color: "white",
      padding: "12px",
      width: "100%",
      marginTop: "16px",
      borderRadius: "8px",
      border: "none",
      cursor: "pointer"
    },
    resultsSection: {
      marginTop: "32px",
      backgroundColor: "white",
      padding: "16px",
      borderRadius: "8px",
      border: "1px solid #fca5cf"
    },
    resultsTitle: {
      fontSize: "20px",
      fontWeight: "600",
      marginBottom: "16px",
      color: "#374151"
    },
    bestMatch: {
      marginBottom: "24px",
      padding: "16px",
      border: "1px solid #d1d5db",
      borderRadius: "8px",
      backgroundColor: "#fdf2f8"
    },
    bestMatchTitle: {
      fontWeight: "bold",
      color: "#db2777",
      marginBottom: "8px"
    },
    imageContainer: {
      width: "100%",
      marginTop: "8px"
    },
    memeImage: {
      width: "100%",
      maxHeight: "240px",
      objectFit: "cover",
      borderRadius: "8px"
    },
    similarity: {
      marginTop: "8px",
      color: "#6b7280"
    },
    similarTitle: {
      fontWeight: "bold",
      color: "#db2777",
      marginBottom: "12px"
    },
    grid: {
      display: "grid",
      gridTemplateColumns: "repeat(2, 1fr)",
      gap: "16px"
    },
    gridItem: {
      padding: "12px",
      border: "1px solid #d1d5db",
      borderRadius: "8px",
      backgroundColor: "#f9fafb"
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.heading}>Looking for a meme?</h1>
        
        {/* Search Input */}
        <input
          type="text"
          placeholder="Search memes..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={styles.searchInput}
        />
        <button 
          onClick={handleSearch} 
          style={styles.searchButton}
        >
          Search
        </button>
        
        {/* Upload Section */}
        <div style={styles.uploadSection}>
          <h2 style={styles.uploadTitle}>Upload a new meme</h2>
          <input 
            type="file" 
            onChange={(e) => setFile(e.target.files[0])} 
            style={styles.fileInput}
          />
          <button 
            onClick={handleUpload} 
            style={styles.uploadButton}
          >
            Upload Meme
          </button>
        </div>
        
        {/* Display Results */}
        {results && (
          <div style={styles.resultsSection}>
            <h2 style={styles.resultsTitle}>Results</h2>
            
            {results.best_match && (
              <div style={styles.bestMatch}>
                <p style={styles.bestMatchTitle}>Best Match</p>
                <div style={styles.imageContainer}>
                  <img 
                    src={`http://localhost:8000/memes/${results.best_match.meme}`} 
                    alt="Best Match" 
                    style={styles.memeImage}
                  />
                </div>
                <p style={styles.similarity}>Similarity: {results.best_match.similarity.toFixed(2)}</p>
              </div>
            )}
            
            {results.similar_memes && results.similar_memes.length > 0 && (
              <div>
                <p style={styles.similarTitle}>Similar Memes</p>
                <div style={styles.grid}>
                  {results.similar_memes.map((m, idx) => (
                    <div key={idx} style={styles.gridItem}>
                      <div style={styles.imageContainer}>
                        <img 
                          src={`http://localhost:8000/memes/${m.meme}`} 
                          alt="Meme" 
                          style={styles.memeImage}
                        />
                      </div>
                      <p style={styles.similarity}>Similarity: {m.similarity.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}