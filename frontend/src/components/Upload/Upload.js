import React from 'react';
import { styles } from '../../styles/theme';

const Upload = ({ onFileChange, onUpload }) => {
  return (
    <div style={styles.uploadSection}>
      <h2 style={styles.uploadTitle}>Upload a New Meme</h2>
      <div style={{ marginBottom: '16px' }}>
        <input 
          type="file" 
          onChange={onFileChange} 
          style={styles.fileInput}
        />
      </div>
      <button 
        onClick={onUpload} 
        style={styles.button}
      >
        Upload
      </button>
    </div>
  );
};

export default Upload; 