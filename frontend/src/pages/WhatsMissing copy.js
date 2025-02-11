import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import ReactMarkdown from 'react-markdown';

const WhatsMissing = () => {
  const [file, setFile] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  const handleTextChange = (event) => {
    setTextInput(event.target.value);
  };

  const handleSubmit = async () => {
    if (!file || !textInput.trim()) {
      alert('Please provide both a file and text input');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('text_input', textInput);

    try {
      const response = await axios.post('http://localhost:8000/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data.result);
    } catch (error) {
      console.error('Error processing request:', error);
      setResult('An error occurred while processing your request.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="title mb-5">What's Missing?</h1>
      <div className="row justify-content-center">
        {/* File Upload Section */}
        <div className="col-md-4">
          <div className="card p-4" style={{ backgroundColor: '#f8f9fa' }}>
            <h5 className="mb-3">Upload DRHP</h5>
            <div className="mb-3">
              <input
                type="file"
                className="form-control"
                onChange={handleFileChange}
                accept=".pdf,.doc,.docx,.txt"
              />
              {fileName && (
                <div className="mt-2 text-muted">
                  Selected file: {fileName}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Text Input Section */}
        <div className="col-md-4">
          <div className="card p-4" style={{ backgroundColor: '#f8f9fa' }}>
            <h5 className="mb-3">Enter Text</h5>
            <div className="mb-3">
              <textarea
                className="form-control"
                rows="5"
                value={textInput}
                onChange={handleTextChange}
                placeholder="Enter your text here..."
              />
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="col-md-8 mt-4 text-center">
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!file || !textInput.trim() || isLoading}
          >
            {isLoading ? 'Processing...' : 'Process'}
          </button>
        </div>

        {/* Results Section */}
        {result && (
          <div className="col-md-8 mt-4">
            <div className="card p-4" style={{ backgroundColor: '#f8f9fa' }}>
              <h5 className="mb-3">Results</h5>
              <ReactMarkdown>{result}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WhatsMissing;
