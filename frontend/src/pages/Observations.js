import React, { useState, useEffect } from 'react';
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "../App.css";
import * as XLSX from 'xlsx';

function Observations() {
    const [cacheRules, setCacheRules] = useState('');
    const [prevObservations, setPrevObservations] = useState('');
    const [pdfFile, setPdfFile] = useState(null);
    const [excelFile, setExcelFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [qaResults, setQaResults] = useState([]);
    const [activeTab, setActiveTab] = useState("");
    const [expanded, setExpanded] = useState(false);
    const [pdfUploaded, setPdfUploaded] = useState(false);
    const [excelUploaded, setExcelUploaded] = useState(false);

    const handleCacheRulesChange = (e) => {
        setCacheRules(e.target.value);
    };

    const handlePrevObservationsChange = (e) => {
        setPrevObservations(e.target.value);
    };

    const handlePdfChange = (e) => {
        const selectedFile = e.target.files[0];
        setPdfFile(selectedFile);
        setPdfUploaded(selectedFile != null);
    };

    const handleExcelChange = (e) => {
        const selectedFile = e.target.files[0];
        setExcelFile(selectedFile);
        setExcelUploaded(selectedFile != null);
    };

    const validateExcelHeaders = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                  const data = new Uint8Array(e.target.result);
                  const workbook = XLSX.read(data, { type: 'array' });
                    const sheetName = workbook.SheetNames[0];
                    const worksheet = workbook.Sheets[sheetName];
                    const headers = XLSX.utils.sheet_to_json(worksheet, { header: 1 })[0];


                    if (
                        !headers ||
                        headers.length !== 5 ||
                        headers[0] !== "questions" ||
                        headers[1] !== "sub_cat_1" ||
                        headers[2] !== "sub_cat_2" ||
                        headers[3] !== "range_c1" ||
                        headers[4] !== "range_c2"
                    ) {
                      reject("Invalid XLSX file header format. Expected: 'questions,sub_cat_1,sub_cat_2,range_c1,range_c2'");
                    } else {
                        resolve();
                    }
                } catch (error) {
                    reject("Error reading or parsing XLSX file.");
                }
            };
             reader.onerror = () => {
                reject("Error reading the XLSX file.");
            };
            reader.readAsArrayBuffer(file);

        });
    };

   const handleFileUpload = async () => {
    if (!pdfFile) {
        toast.error("Please upload a PDF file.", { position: "top-right", autoClose: 3000 });
        return;
    }
    if (!excelFile) {
        toast.error("Please upload an XLSX file.", { position: "top-right", autoClose: 3000 });
        return;
    }

    setLoading(true);
    try {
        await validateExcelHeaders(excelFile)

        const formData = new FormData();
        formData.append("pdf", pdfFile);
        formData.append("excel_file", excelFile); // Corrected FormData field name

        toast.info("Processing...", { position: "top-right", autoClose: 7000 });
        const response = await axios.post(
            `http://localhost:8000/observations/process_v2`,
            formData,
            {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            }
        );

        console.log("Response data:", response.data);
        if (response.data && response.data.qa_results) {
            setQaResults(response.data.qa_results);
        }

        toast.success("Files processed successfully", {
            position: "top-right",
            autoClose: 3000,
        });

    } catch (error) {
        console.error("Full error response:", error.response); // Full error logging
        toast.error(
            error.response?.data?.detail || error || "Error processing files",
            { position: "top-right", autoClose: 3000 }
        );
    } finally {
        setLoading(false);
    }
};



  const groupedResults = qaResults.reduce((acc, item) => {
        const section = item.sub_category || "Other";
        if (!acc[section]) {
            acc[section] = [];
        }
        acc[section].push(item);
        return acc;
    }, {});

    useEffect(() => {
        if (qaResults.length > 0) {
            const firstSection = Object.keys(groupedResults)[0];
            setActiveTab(firstSection);
        }
    }, [qaResults]);

  const toggleExpandAll = () => {
    setExpanded(!expanded);
  };

  return (
    <div className="container-fluid col-lg-11">
      <div className="page mx-auto pt-4">
        <h1 className="title">DRHP Processing - Observations</h1>
        <div className="info alert alert-info mt-4">
            Users may upload a DRHP (PDF) and an Excel file (.xlsx) containing questions and subcategories. The module will process the DRHP based on the provided Excel file and provide observations.
            <br />
            <strong>Ensure your Excel file has the following headers:</strong>
            <br />
            <code>questions,sub_cat_1,sub_cat_2,range_c1,range_c2</code>
        </div>
        <div className="file-upload mb-4 mt-2">
          <div className="mb-3">
              <input
                  type="file"
                  id="pdf-input"
                  onChange={handlePdfChange}
                  className="form-control-file"
                  accept=".pdf"
              />
                <label htmlFor="pdf-input" className="mt-2">
                    Drag and drop DRHP PDF file here or click to browse
                </label>
                 {pdfUploaded && <div className="text-success mt-1">PDF file uploaded</div>}
          </div>
<br></br>
<hr></hr>
<br></br>
            <div className="mb-3">
                <input
                    type="file"
                    id="excel-input"
                    onChange={handleExcelChange}
                    className="form-control-file"
                    accept=".xlsx, .xls"
                />
                <label htmlFor="excel-input" className="mt-2">
                    Drag and drop Questions XLSX file here or click to browse
                </label>
                {excelUploaded && <div className="text-success mt-1">XLSX file uploaded</div>}
             </div>
<br></br>
<hr></hr>
<br></br>
            <button
                onClick={handleFileUpload}
                className="btn btn-primary mt-3"
                disabled={loading}
            >
                Process Files
            </button>
             <div className="mb-3 mt-3">
                <label htmlFor="cache-rules" className="form-label">Regulation Rules (optional)</label>
                <textarea
                    className="form-control"
                    id="cache-rules"
                    rows="3"
                    value={cacheRules}
                    onChange={handleCacheRulesChange}
                    placeholder="Enter cache rules here (optional)"
                />
            </div>

            <div className="mb-3 mt-3">
                <label htmlFor="prev-observations" className="form-label">Previous Observations (Optional)</label>
                <textarea
                    className="form-control"
                    id="prev-observations"
                    rows="3"
                    value={prevObservations}
                    onChange={handlePrevObservationsChange}
                    placeholder="Enter previous observations here (optional)"
                />
            </div>
        </div>

        {/* Loading Spinner */}
        {loading && (
          <div className="text-center mt-4">
            <div className="spinner-border" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        )}

        {/* Show the Q&A in tabbed format */}
        {!loading && qaResults.length > 0 && (
          <div className="row mt-4">
            <div className="col-md-12 mt-2">
              <div className="d-flex justify-content-between mb-4">
                <h2 className="h4">Observations</h2>
                <button className="btn btn-secondary" onClick={toggleExpandAll}>
                  {expanded ? "Collapse All" : "Expand All"}
                </button>
              </div>

              {/* Tabs */}
              <div
                className="nav nav-tabs mb-4"
                style={{
                  borderBottom: "2px solid #e2e8f0",
                  display: "flex",
                  overflowX: "auto",
                  whiteSpace: "nowrap",
                }}
              >
                {Object.keys(groupedResults).map((section, index) => (
                  <button
                    key={index}
                    className={`nav-link ${
                      activeTab === section ? "active" : ""
                    }`}
                    onClick={() => setActiveTab(section)}
                    style={{
                      padding: "12px 20px",
                      border: "1px solid #282c34",
                      marginBottom: "-2px",
                      cursor: "pointer",
                      transition: "all 0.2s",
                    }}
                  >
                    {section}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="tab-content">
                {Object.entries(groupedResults).map(([section, items]) => (
                  <div
                    key={section}
                    style={{
                      display: activeTab === section ? "block" : "none",
                    }}
                  >
                    {items.map((item, index) => (
                      <div className="mb-3" key={index}>
                        <div
                          className="accordion"
                          id={`accordionExample${index}`}
                        >
                          <div className="accordion-item">
                            <h2
                              className="accordion-header"
                              id={`heading${index}`}
                            >
                              <button
                                className="accordion-button"
                                type="button"
                                data-bs-toggle="collapse"
                                data-bs-target={`#collapse${index}`}
                                aria-expanded={expanded}
                                aria-controls={`collapse${index}`}
                              >
                                <strong>
                                  {item.title || `Query ${index + 1}`}
                                </strong>
                              </button>
                            </h2>
                            <div
                              id={`collapse${index}`}
                              className="accordion-collapse collapse"
                              aria-labelledby={`heading${index}`}
                              data-bs-parent={`#accordionExample${index}`}
                            >
                              <div
                                className="accordion-body"
                                style={{ textAlign: "left" }}
                              >
                                <p>
                                  <strong>Question:</strong> {item.question}
                                </p>
                                <hr />
                                <p>
                                  <strong>Answer:</strong>
                                </p>
                                <div
                                  style={{
                                    whiteSpace: "pre-wrap",
                                    wordWrap: "break-word",
                                    textAlign: "left",
                                  }}
                                >
                                  {item.answer}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
         <ToastContainer />
      </div>
    </div>
  );
}

export default Observations;
