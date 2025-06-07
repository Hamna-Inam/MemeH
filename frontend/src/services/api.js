const API_BASE_URL = 'http://localhost:8000';

export const searchMemes = async (query) => {
  try {
    const response = await fetch(`${API_BASE_URL}/search?query=${encodeURIComponent(query)}&threshold=0.25&k=1`);
    
    if (!response.ok) {
      throw new Error(`Search failed: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
};

export const uploadMeme = async (file) => {
  try {
    // Validate file before uploading
    if (!file) {
      throw new Error('No file provided');
    }
    
    // Check file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload a JPEG or PNG image.');
    }
    
    // Check file size 
    const maxSizeInMB = 10;
    if (file.size > maxSizeInMB * 1024 * 1024) {
      throw new Error(`File too large. Maximum size is ${maxSizeInMB}MB.`);
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    console.log('Uploading file:', file.name, file.type, file.size);
    
    const response = await fetch(`${API_BASE_URL}/upload/`, {
      method: 'POST',
      body: formData,
    });
    
    const responseData = await response.json();
    
    if (!response.ok) {
      // Handle specific error from backend
      const errorMessage = responseData.error || `Upload failed: ${response.status} ${response.statusText}`;
      throw new Error(errorMessage);
    }
    
    console.log('Upload successful:', responseData);
    return responseData; // Return the full response data
    
  } catch (error) {
    console.error('Upload error:', error);
    throw error; // Re-throw so the calling code can handle it
  }
};

export const getMemeUrl = (memePath) => {
  return `${API_BASE_URL}/memes/${memePath}`;
};
