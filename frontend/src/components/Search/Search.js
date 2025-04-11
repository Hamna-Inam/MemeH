import React from 'react';
import { styles } from '../../styles/theme';

const Search = ({ query, setQuery, onSearch, onFileChange, onUpload }) => {
  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      onFileChange(e);
      onUpload();
    }
  };

  return (
    <div style={{ 
      display: 'flex',
      justifyContent: 'center',
      marginBottom: '32px'
    }}>
      <div style={{ 
        display: 'inline-flex',
        gap: '8px',
        height: '50px',
        maxWidth: '600px'
      }}>
        <div style={{ 
          display: 'flex',
          alignItems: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '8px',
          minWidth: '400px'
        }}>
          <input
            type="text"
            placeholder="Search memes..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              ...styles.searchInput,
              flex: 1,
              height: '50px',
              lineHeight: '50px',
              paddingLeft: '16px',
              paddingRight: '16px',
              backgroundColor: 'transparent',
              border: 'none',
              fontSize: '16px',
              color: 'rgba(255, 255, 255, 0.9)',
              outline: 'none'
            }}
          />
        </div>
        <button 
          onClick={onSearch} 
          style={{
            height: '50px',
            padding: '0 16px',
            width: '80px',
            backgroundColor: '#ec4899',
            borderRadius: '8px',
            border: 'none',
            color: 'white',
            fontSize: '14px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          Search
        </button>
        <div>
          <input
            type="file"
            onChange={handleFileChange}
            style={{
              display: 'none'
            }}
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '50px',
              padding: '0 16px',
              width: '80px',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '8px',
              border: 'none',
              color: 'white',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            Upload
          </label>
        </div>
      </div>
    </div>
  );
};

export default Search; 