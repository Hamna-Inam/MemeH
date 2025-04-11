import React from 'react';
import { getMemeUrl } from '../../services/api';

const Results = ({ results }) => {
  if (!results) return null;

  return (
    <div>
      {results.message && (
        <div style={{
          color: 'rgba(255, 255, 255, 0.9)',
          fontSize: '18px',
          marginBottom: '24px',
          textAlign: 'center'
        }}>
          {results.message}
        </div>
      )}
      
      <div style={{
        marginTop: '32px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '24px'
      }}>
        {results.best_match && (
          <div style={{
            width: '100%',
            borderRadius: '8px',
            overflow: 'hidden'
          }}>
            <img 
              src={getMemeUrl(results.best_match.meme)} 
              alt="Best Match" 
              style={{
                width: '100%',
                height: 'auto',
                objectFit: 'contain'
              }}
            />
          </div>
        )}
        
        {results.similar_memes && results.similar_memes.length > 0 && (
          results.similar_memes.map((m, idx) => (
            <div key={idx} style={{
              width: '100%',
              borderRadius: '8px',
              overflow: 'hidden'
            }}>
              <img 
                src={getMemeUrl(m.meme)} 
                alt="Meme" 
                style={{
                  width: '100%',
                  height: 'auto',
                  objectFit: 'contain'
                }}
              />
            </div>
          ))
        )}
      </div>

      {results.upload_suggestion && (
        <div style={{
          color: 'rgba(255, 255, 255, 0.7)',
          fontSize: '16px',
          marginTop: '24px',
          textAlign: 'center'
        }}>
          {results.upload_suggestion}
        </div>
      )}
    </div>
  );
};

export default Results; 