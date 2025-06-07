import React from 'react';
import { styles } from '../../styles/theme';

const Upload = ({ onFileChange, onUpload, uploading, file }) => {
  return (
    <div style={styles.uploadSection}>
      <h2 style={styles.uploadTitle}>Upload a New Meme</h2>
      <div style={{ marginBottom: '16px' }}>
        <input
          type="file"
          onChange={onFileChange}
          accept="image/*"
          style={styles.fileInput}
          disabled={uploading}
        />
      </div>
      <button
        onClick={onUpload}
        style={{
          ...styles.button,
          backgroundColor: uploading ? '#9ca3af' : styles.button.backgroundColor,
          cursor: uploading ? 'not-allowed' : 'pointer'
        }}
        disabled={uploading || !file}
      >
        {uploading ? 'Uploading...' : 'Upload'}
      </button>
    </div>
  );
};

export default Upload;