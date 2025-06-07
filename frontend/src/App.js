import React, { useState } from "react";
import { styles } from "./styles/theme";
import Search from "./components/Search/Search";
import Results from "./components/Results/Results";
import { searchMemes } from "./services/api";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleSearch = async () => {
    try {
      const data = await searchMemes(query);
      setResults(data);
    } catch (error) {
      console.error('Search error:', error);
      alert(`Search failed: ${error.message}`);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      // Keep the fallback reference that was making it work
      window.currentFile = selectedFile;
    } else {
      setFile(null);
      window.currentFile = null;
    }
  };

  const handleUpload = async () => {
    // Use the working logic with fallback
    const fileToUpload = file || window.currentFile;
    
    if (!fileToUpload) {
      alert("Please select a file first!");
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", fileToUpload);
      
      const response = await fetch("http://localhost:8000/upload/", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        alert("Meme uploaded successfully!");
        setFile(null);
        window.currentFile = null;
        const fileInput = document.querySelector('input[type="file"]');
        if (fileInput) fileInput.value = '';
      } else {
        throw new Error(`Upload failed with status: ${response.status}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert(`Upload failed: ${error.message || 'Unknown error'}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <img
        src="/willl.png"
        alt="Will Smith"
        style={{
          position: "fixed",
          top: "50%",
          right: "20px",
          transform: "translateY(-50%) scaleX(-1)",
          height: "700px",
          zIndex: 0,
          opacity: 0.9,
          pointerEvents: "none"
        }}
      />
      
      <div style={styles.container}>
        <div style={styles.card}>
          <h1 style={styles.heading}>Welcome to Skibidi Rizz Search Engine!</h1>
          
          <Search
            query={query}
            setQuery={setQuery}
            onSearch={handleSearch}
            onFileChange={handleFileChange}
            onUpload={handleUpload}
            uploading={uploading}
            file={file}
          />
          
          <Results results={results} />
        </div>
      </div>
    </div>
  );
}

export default App;