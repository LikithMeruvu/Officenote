import React, { useState } from 'react';
import axios from 'axios';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../App.css';
import * as XLSX from 'xlsx';

function DBCheck() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const validTypes = ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel', 'text/csv'];
      if (!validTypes.includes(selectedFile.type)) {
        toast.error('Please upload only Excel or CSV files', {
          position: 'top-right',
          autoClose: 3000
        });
        return;
      }
      setFile(selectedFile);
      setResults([]); // Clear previous results when a new file is uploaded
      setLoading(true);

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        toast.info('Processing...', { position: 'top-right', autoClose: 7000 });
        const response = await axios.post('http://localhost:8000/db-check', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        setResults(response.data.matches);
        toast.success('File processed successfully', {
          position: 'top-right',
          autoClose: 3000
        });
      } catch (error) {
        console.error('Error processing file:', error);
        toast.error('Error processing file', {
          position: 'top-right',
          autoClose: 3000
        });
      } finally {
        setLoading(false);
      }
    }
  };

  const handleDownload = () => {
    if (results.length === 0) {
      toast.warning('No data to download', {
        position: 'top-right',
        autoClose: 3000
      });
      return;
    }

    // Create a new workbook
    const wb = XLSX.utils.book_new();

    results.forEach((result, index) => {
      // Prepare data for each Excel sheet
      const excelData = [{
        'Matched Name': result.matched_user_name,
        'File Name': result.excel_file_name,
        ...result.row_data
      }];

      // Create worksheet for each result
      const ws = XLSX.utils.json_to_sheet(excelData);

      // Use generic sheet name
      const sheetName = `Sheet${index + 1}`;

      XLSX.utils.book_append_sheet(wb, ws, sheetName);
    });

    // Generate and download file
    XLSX.writeFile(wb, 'database_check_results.xlsx');
  };

  return (
    <div className="container-fluid col-lg-11">
      <ToastContainer />
      <div className="page mx-auto pt-4">
        <h1 className="title">DRHP Processing - Database Check</h1>
        <div className='info alert alert-info mt-4'>
          Users may upload an Excel or CSV file with names to be checked against the list of Prosecution cases available at SEBI Website <a href='https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes&cid=14' target='_blank' rel='noopener noreferrer'>(Click Here)</a> or the list or the Vanishing companies.
        </div>

        <div className="file-upload mb-4 mt-2">
          <input
            type="file"
            id="file-input"
            onChange={handleFileChange}
            className="form-control-file"
            accept=".xlsx,.xls,.csv"
          />
          <label htmlFor="file-input" className="mt-2">
            Drag and drop an Excel / CSV file here or click to browse
          </label>
        </div>

        {/* Loading Spinner */}
        {loading && (
          <div className="text-center mt-4">
            <div className="spinner-border" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        )}

        {results.length > 0 && (
          <div className="mt-4">
            <button 
              className="btn btn-primary mb-4" 
              onClick={handleDownload}
              style={{ width: "180px" }}
            >
              Download Results
            </button>

            {results.map((result, index) => (
              <div key={index} className="card mb-4 shadow-sm">
                <div 
                  className="card-header"
                  style={{ 
                    backgroundColor: '#f8fafc',
                    borderBottom: '1px solid #e2e8f0'
                  }}
                >
                  <strong>Match #{index + 1}: {result.matched_user_name}</strong>
                </div>
                <div className="card-body">
                  <div className="row mb-3">
                    <div className="col-md-6">
                      <strong>File Name:</strong> {result.excel_file_name}
                    </div>
                    <div className="col-md-6 text-md-end">
                      <strong>File Link:</strong> <a href={result.pdf_link} target="_blank" rel="noopener noreferrer">Open PDF</a>
                    </div>
                  </div>
                  <div>
                    <strong>Details:</strong>
                    <div className="table-responsive mt-2">
                      <table className="table table-bordered">
                        <thead className="table-light">
                          <tr>
                            {Object.keys(result.row_data).map((key, idx) => (
                              <th key={idx}>{key}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            {Object.values(result.row_data).map((value, idx) => (
                              <td key={idx}>{value || '-'}</td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default DBCheck;
