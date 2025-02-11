import React, { useState } from 'react';
import axios from 'axios';

const WhatsMissing = () => {
  const [companyName, setCompanyName] = useState('');
  const [personName, setPersonName] = useState('');
  const [role, setRole] = useState('');
  const [drhpFile, setDrhpFile] = useState(null);
  const [queryResults, setQueryResults] = useState([]);
  const [allLitigations, setAllLitigations] = useState([]);
  const [verificationResponse, setVerificationResponse] = useState('');
  const [showAccordion, setShowAccordion] = useState(false);
  const [companyNameLocked, setCompanyNameLocked] = useState(false);
  const [loading, setLoading] = useState(false);
  const [whatsMissingLoading, setWhatsMissingLoading] = useState(false);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  const handleDrhpUpload = (event) => {
    setDrhpFile(event.target.files[0]);
  };

  const handleCompanyNameChange = (event) => {
    if (!companyNameLocked) {
      setCompanyName(event.target.value);
    }
  };

  const handlePersonNameChange = (event) => {
    setPersonName(event.target.value);
  };

  const handleRoleChange = (event) => {
    setRole(event.target.value);
  };

  const handleQuery = async () => {
    if (!companyName || !personName || !role) {
      alert('Please fill in all fields before querying');
      return;
    }
    setLoading(true);
    try {
      if (!companyNameLocked) {
        setCompanyNameLocked(true);
      }
      const formData = new FormData();
      formData.append('company_name', companyName);
      formData.append('person_name', personName);
      formData.append('role', role);

      const response = await axios.post(
        'http://localhost:8000/process_perplexity',
        formData,
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }
      );

      setQueryResults((prevResults) => [
        ...prevResults,
        {
          personName: personName,
          role: role,
          perplexityResponses: response.data.perplexity_responses,
          foundLitigations: response.data.found_litigations,
        },
      ]);
      setAllLitigations((prevLitigations) => [
        response.data.found_litigations,
        ...prevLitigations,
      ]);
      setShowAdvancedOptions(true);
    } catch (error) {
      console.error('Error calling /process_perplexity:', error);
      alert('Error calling /process_perplexity. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const formatLitigations = () => {
    return allLitigations
      .map((litigation, index) => `${index + 1}. ${litigation}`)
      .join('\n');
  };

  const handleWhatsMissing = async () => {
    if (!drhpFile) {
      alert('You need to upload DRHP document of selected company');
      return;
    }
    setWhatsMissingLoading(true);
    try {
      const formattedLitigations = formatLitigations();

      const formData = new FormData();
      formData.append('drhp_pdf', drhpFile);
      formData.append('litigations', formattedLitigations);

      const response = await axios.post(
        'http://localhost:8000/process',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setVerificationResponse(
        response.data[0]?.verification_response ||
          'No verification response received.'
      );
      setShowAccordion(true);
    } catch (error) {
      console.error('Error calling /process:', error);
      alert('Error calling /process. Check console for details.');
    } finally {
      setWhatsMissingLoading(false);
    }
  };

  return (
    <div className="container-fluid col-lg-11">
      <div className="page mx-auto pt-4">
        <h1 className="title">DRHP Processing - What's Missing</h1>
        <div className="info alert alert-info mt-4">
          Users may enter details about a person and their role to check for any missing information or disclosures in the DRHP.
        </div>

        {/* Input Form */}
        <div className="card shadow-sm mt-4">
          <div 
            className="card-header"
            style={{ 
              backgroundColor: '#f8fafc',
              borderBottom: '1px solid #e2e8f0'
            }}
          >
            <strong>Search Details</strong>
          </div>
          <div className="card-body">
            <div className="row g-3">
              {/* Company Name */}
              <div className="col-md-4">
                <label className="form-label">Company Name</label>
                <input
                  type="text"
                  className="form-control"
                  value={companyName}
                  onChange={handleCompanyNameChange}
                  placeholder="Enter company name"
                  disabled={companyNameLocked}
                />
              </div>

              {/* Person Name */}
              <div className="col-md-4">
                <label className="form-label">Person Name</label>
                <input
                  type="text"
                  className="form-control"
                  value={personName}
                  onChange={handlePersonNameChange}
                  placeholder="Enter person name"
                />
              </div>

              {/* Role */}
              <div className="col-md-4">
                <label className="form-label">Role</label>
                <input
                  type="text"
                  className="form-control"
                  value={role}
                  onChange={handleRoleChange}
                  placeholder="Enter role"
                />
              </div>

              {/* Query Button */}
              <div className="col-12">
                <button
                  onClick={handleQuery}
                  disabled={loading}
                  className="btn btn-primary"
                  style={{ width: "120px" }}
                >
                  {loading ? (
                    <div className="spinner-border spinner-border-sm" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                  ) : (
                    'Query'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Results Section */}
        {queryResults.length > 0 && (
          <div className="mt-4">
            {queryResults.map((result, index) => (
              <div key={index} className="card shadow-sm mb-4">
                <div 
                  className="card-header d-flex justify-content-between align-items-center"
                  style={{ 
                    backgroundColor: '#f8fafc',
                    borderBottom: '1px solid #e2e8f0'
                  }}
                >
                  <strong>{result.personName} ({result.role})</strong>
                </div>
                <div className="card-body">
                  {result.perplexityResponses.map((response, rIndex) => (
                    <div key={rIndex} className="mb-4">
                      <h5 className="mb-3">{response.query}</h5>
                      <div className="p-3 bg-light rounded">
                        <p style={{ whiteSpace: 'pre-wrap' }}>
                          {response.response.choices[0].message.content}
                        </p>
                        {response.response.citations && (
                          <div className="mt-3">
                            <strong>Citations:</strong>
                            <ul className="mt-2">
                              {response.response.citations.map((citation, cIndex) => (
                                <li key={cIndex}>
                                  <a
                                    href={citation}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-primary"
                                  >
                                    {citation}
                                  </a>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {result.foundLitigations && (
                    <div className="mt-4 p-3 bg-light rounded">
                      <h5 className="mb-3">Found Litigations:</h5>
                      <p style={{ whiteSpace: 'pre-wrap' }}>{result.foundLitigations}</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Loading Spinner */}
        {loading && (
          <div className="text-center mt-4">
            <div className="spinner-border" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        )}

        {showAdvancedOptions && (
          <>
            <div
              className="card shadow-sm mt-4"
            >
              <div 
                className="card-header"
                style={{ 
                  backgroundColor: '#f8fafc',
                  borderBottom: '1px solid #e2e8f0'
                }}
              >
                <strong>Upload DRHP PDF</strong>
              </div>
              <div className="card-body">
                <input
                  type="file"
                  onChange={handleDrhpUpload}
                  accept=".pdf"
                />
              </div>
            </div>

            <div
              className="text-center mt-4"
            >
              <button
                onClick={handleWhatsMissing}
                disabled={whatsMissingLoading || !drhpFile}
                className="btn btn-primary"
              >
                {whatsMissingLoading ? (
                  <div className="spinner-border spinner-border-sm" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                ) : (
                  "What's Missing?"
                )}
              </button>
            </div>

            {showAccordion && (
              <div className="card shadow-sm mt-4">
                <div 
                  className="card-header"
                  style={{ 
                    backgroundColor: '#f8fafc',
                    borderBottom: '1px solid #e2e8f0'
                  }}
                >
                  <strong>Verification Response</strong>
                </div>
                <div className="card-body">
                  <p style={{ whiteSpace: 'pre-wrap' }}>{verificationResponse}</p>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default WhatsMissing;
