const API_BASE_URL = 'http://localhost:8000';

export const searchMemes = async (query) => {
  const response = await fetch(`${API_BASE_URL}/search?query=${query}&threshold=0.25&k=3`);
  return await response.json();
};

export const uploadMeme = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/upload/`, {
    method: 'POST',
    body: formData,
  });
  
  return response.ok;
};

export const getMemeUrl = (memePath) => {
  return `${API_BASE_URL}/memes/${memePath}`;
}; 