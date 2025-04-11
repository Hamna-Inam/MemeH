import React, { useState } from "react";
import { styles } from "./styles/theme";
import Search from "./components/Search/Search";
import Results from "./components/Results/Results";
import { searchMemes, uploadMeme } from "./services/api";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [file, setFile] = useState(null);

  const handleSearch = async () => {
    const data = await searchMemes(query);
    setResults(data);
  };

  const handleUpload = async () => {
    if (!file) return;
    const success = await uploadMeme(file);
    if (success) {
      alert("Meme uploaded successfully!");
      setFile(null); // Reset file after upload
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.heading}>Welcome to Skibidi Rizz Search Engine!</h1>
        
        <Search 
          query={query}
          setQuery={setQuery}
          onSearch={handleSearch}
          onFileChange={(e) => setFile(e.target.files[0])}
          onUpload={handleUpload}
        />
        
        <Results results={results} />
      </div>
    </div>
  );
}

export default App;