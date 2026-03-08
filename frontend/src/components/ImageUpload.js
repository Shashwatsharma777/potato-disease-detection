import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import '../styles/ImageUpload.css';

function ImageUpload({ onDrop }) {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrop = useCallback((acceptedFiles) => {
    setIsDragActive(false);
    onDrop(acceptedFiles);
  }, [onDrop]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: handleDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp'],
    },
    multiple: false,
  });

  const handleDragEnter = () => {
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  return (
    <div className="upload-container">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
      >
        <input {...getInputProps()} />
        <p>🌿 Drag and drop a leaf image here</p>
        <p className="or-text">Potato · Tomato · Bell Pepper</p>
        <p className="or-text">or click to select a file</p>
      </div>
    </div>
  );
}

export default ImageUpload;
