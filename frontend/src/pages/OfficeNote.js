import React, { Component, useState } from "react";
import axios from "axios";
import { saveAs } from "file-saver";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import "bootstrap/dist/css/bootstrap.min.css";
import "../App.css";

function OfficeNote() {
  // State variables
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState("");
  const [pdfUrl, setPdfUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [extractedInfo, setExtractedInfo] = useState({});

  // Event handlers
  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    if (selectedFile) {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        toast.info("Processing...", { position: "top-right", autoClose: 7000 });
        console.log("Uploading file...");
        const response = await axios.post(
          `http://localhost:8000/upload`,
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        console.log("File uploaded successfully:", response.data);
        setFileUrl(response.data.file_url);
        setExtractedInfo(response.data.extracted_info);
        toast.success("File processed successfully", {
          position: "top-right",
          autoClose: 3000,
        });
      } catch (error) {
        console.error("Error uploading file:", error);
        toast.error("Error processing file", {
          position: "top-right",
          autoClose: 3000,
        });
      } finally {
        setLoading(false);
      }
    }
  };

  const handleUrlChange = (e) => {
    setPdfUrl(e.target.value);
  };

  const handleUrlUpload = async () => {
    if (pdfUrl) {
      setLoading(true);
      try {
        toast.info("Processing...", { position: "top-right", autoClose: 3000 });
        const response = await axios.post(
          `http://localhost:8000/upload_from_url`,
          null,
          {
            params: { url: pdfUrl },
          }
        );

        console.log("PDF from URL processed successfully:", response.data);
        setFileUrl(response.data.file_url);
        setExtractedInfo(response.data.extracted_info);
        toast.success("PDF from URL processed successfully", {
          position: "top-right",
          autoClose: 3000,
        });
      } catch (error) {
        console.error("Error processing URL:", error);
        toast.error("Error processing URL", {
          position: "top-right",
          autoClose: 3000,
        });
      } finally {
        setLoading(false);
      }
    }
  };

  const handleInputChange = (e) => {
    const { id, value } = e.target;
    setExtractedInfo((prevInfo) => ({
      ...prevInfo,
      [id]: value,
    }));
  };

  const handleUpdate = () => {
    console.log("Updated extracted info:", extractedInfo);
    toast.success("Info updated successfully!", {
      position: "top-right",
      autoClose: 3000,
    });
  };

  const handleGenerateOfficeNote = async () => {
    try {
      const response = await axios.post(
        `http://localhost:8000/generate_office_note`,
        extractedInfo,
        {
          responseType: "blob",
        }
      );

      const blob = new Blob([response.data], {
        type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      });

      saveAs(blob, "GeneratedOfficeNote.docx");
      toast.success("Office note generated successfully", {
        position: "top-right",
        autoClose: 3000,
      });
    } catch (error) {
      console.error("Error generating office note:", error);
      toast.error("Error generating office note", {
        position: "top-right",
        autoClose: 3000,
      });
    }
  };

  // Render
  return (
    <div className="container-fluid col-lg-11">
      <div className="page mx-auto pt-4">
        <h1 className="title">DRHP Processing - Office Note</h1>
        <div className='info alert alert-info mt-4'>
          Users may upload a DRHP and required information for Office Note will be extracted from it into an editable form. The Note may be downloaded as a MS Word document.
        </div>

        <div className="file-upload mb-4 mt-2">
          <input
            type="file"
            id="file-input"
            onChange={handleFileChange}
            className="form-control-file"
            accept=".pdf"
          />
          <label htmlFor="file-input" className="mt-2">
            Drag and drop a file here or click to browse
          </label>
        </div>

        {fileUrl && (
          <div className="row mt-4">
            <div className="col-md-6">
              <div className="card shadow-sm mb-4">
                <div 
                  className="card-header"
                  style={{ 
                    backgroundColor: '#f8fafc',
                    borderBottom: '1px solid #e2e8f0'
                  }}
                >
                  <strong>PDF Preview</strong>
                </div>
                <div className="card-body p-0">
                  <iframe
                    src={fileUrl}
                    width="100%"
                    height="675px"
                    title="PDF Preview"
                    style={{ border: 'none' }}
                  ></iframe>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              {loading ? (
                <div className="text-center mt-4">
                  <div className="spinner-border" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                </div>
              ) : (
                <div className="card shadow-sm">
                  <div 
                    className="card-header"
                    style={{ 
                      backgroundColor: '#f8fafc',
                      borderBottom: '1px solid #e2e8f0'
                    }}
                  >
                    <strong>Office Note Details</strong>
                  </div>
                  <div className="card-body">
                    <div className="table-responsive">
                      <table className="table table-bordered">
                        <tbody>
                          {/* Basic issue details */}
                          <tr>
                            <th style={{ width: '30%', backgroundColor: '#f8fafc' }}>Name of Issuer Company</th>
                            <td>
                              <input
                                type="text"
                                id="issuer_name"
                                value={extractedInfo.issuer_name || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Abbreviation</th>
                            <td>
                              <input
                                type="text"
                                id="abbreviation"
                                value={extractedInfo.abbreviation || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>

                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Lead Managers</th>
                            <td>
                              <textarea
                                type="text"
                                id="lead_manager_list"
                                value={extractedInfo.lead_manager_list || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Date of incorporation</th>
                            <td>
                              <input
                                type="text"
                                id="date_of_incorp"
                                value={extractedInfo.date_of_incorp || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Type of issue</th>
                            <td>
                              <input
                                type="text"
                                id="issue_type"
                                value={extractedInfo.issue_type || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Type of Instrument</th>
                            <td>
                              <input
                                type="text"
                                id="instrument"
                                value={extractedInfo.instrument || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Fixed price/book built</th>
                            <td>
                              <input
                                type="text"
                                id="bidding_type"
                                value={extractedInfo.bidding_type || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Whether underwritten or not</th>
                            <td>
                              <textarea
                                type="text"
                                id="underwriting_agreement"
                                value={extractedInfo.underwriting_agreement || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>

                          {/* Existing paid up share capital */}
                          <tr>
                            <th colSpan="2" style={{ backgroundColor: '#f8fafc' }}>Existing paid up Share capital</th>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Face Value of Shares</th>
                            <td>
                              <input
                                type="text"
                                id="exist_face_value"
                                value={extractedInfo.exist_face_value || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Number of Equity shares</th>
                            <td>
                              <input
                                type="text"
                                id="exist_num_shares"
                                value={extractedInfo.exist_num_shares || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Nominal Amount</th>
                            <td>
                              <input
                                type="text"
                                id="exist_amount"
                                value={extractedInfo.exist_amount || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Existing Share Premium</th>
                            <td>
                              <input
                                type="text"
                                id="exist_share_premium"
                                value={extractedInfo.exist_share_premium || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>

                          {/* Issue Size */}
                          <tr>
                            <th colSpan="2" style={{ backgroundColor: '#f8fafc' }}>Issue Size</th>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Face Value of Shares</th>
                            <td>
                              <input
                                type="text"
                                id="issue_face_value"
                                value={extractedInfo.issue_face_value || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>No. of Shares (Fresh)</th>
                            <td>
                              <input
                                type="text"
                                id="fresh_issue_size"
                                value={extractedInfo.fresh_issue_size || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>No. of Shares (OFS)</th>
                            <td>
                              <input
                                type="text"
                                id="ofs_size"
                                value={extractedInfo.ofs_size || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Nominal Amount (Fresh)</th>
                            <td>
                              <input
                                type="text"
                                id="fresh_issue_amount"
                                value={extractedInfo.fresh_issue_amount || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Nominal Amount (OFS)</th>
                            <td>
                              <input
                                type="text"
                                id="ofs_amount"
                                value={extractedInfo.ofs_amount || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Nominal Amount (Total)</th>
                            <td>
                              <input
                                type="text"
                                id="total_issue_amount"
                                value={extractedInfo.total_issue_amount || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>

                          {/* Post-issue Share Premium and Total Issue Size */}
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Post-issue Share Premium</th>
                            <td>
                              <input
                                type="text"
                                id="post_issue_share_premium"
                                value={extractedInfo.post_issue_share_premium || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Total Issue Size</th>
                            <td>
                              <input
                                type="text"
                                id="total_issue_size"
                                value={extractedInfo.total_issue_size || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>

                          <tr>
                            <th colSpan="2" style={{ backgroundColor: '#f8fafc' }} className="text-center">
                              Post Issue Capital
                            </th>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Face Value</th>
                            <td>
                              <input
                                type="text"
                                id="post_face_value"
                                value={extractedInfo.post_face_value || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Number of shares</th>
                            <td>
                              <input
                                type="text"
                                id="post_shares"
                                value={extractedInfo.post_shares || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Amount (Rs. Crores)</th>
                            <td>
                              <input
                                type="text"
                                id="post_amount"
                                value={extractedInfo.post_amount || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>



                          {/* Eligibility Norms */}
                          <tr>
                            <th colSpan="2" style={{ backgroundColor: '#f8fafc' }}>Other Details</th>
                          </tr>

                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Objects of the issue and funding requirements</th>
                            <td>
                              <textarea
                                type="text"
                                id="objects"
                                value={extractedInfo.objects || ""}
                                className="form-control"
                                onChange={handleInputChange}
                                rows="4"
                              />
                            </td>
                          </tr>

                          {/* Means of finance */}
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Means of finance</th>
                            <td>
                              <input
                                type="text"
                                id="means_finance"
                                value={extractedInfo.means_finance || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>

                          {/* Industry */}
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Industry / Industry group to which issuer belongs to</th>
                            <td>
                              <input
                                type="text"
                                id="industry"
                                value={extractedInfo.industry || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>

                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Exemption from Rule 19(2)(b) of SC(R)R, 1957</th>
                            <td>
                              <input
                                type="text"
                                id="scrr_exemption"
                                value={extractedInfo.scrr_exemption || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Pre issue promoter holding as a % of pre issue paid up capital</th>
                            <td>
                              <input
                                type="text"
                                id="pre_issue_promoter"
                                value={extractedInfo.pre_issue_promoter || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Post issue promoters holding as a % of post issue paid up capital</th>
                            <td>
                              <input
                                type="text"
                                id="post_issue_promoter"
                                value={extractedInfo.post_issue_promoter || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Net Offer to the public as percentage of the total post issue paid up capital</th>
                            <td>
                              <input
                                type="text"
                                id="net_public_offer"
                                value={extractedInfo.net_public_offer || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Lock in of minimum promoters’ contribution</th>
                            <td>
                              <textarea
                                type="text"
                                id="lockin"
                                value={extractedInfo.lockin || ""}
                                className="form-control"
                                onChange={handleInputChange}
                                rows="5"
                              />
                            </td>
                          </tr>

                          {/* Additional Fields not explicitly shown in the images but mentioned */}
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Underwriting Agreement</th>
                            <td>
                              <input
                                type="text"
                                id="underwriting_agreement"
                                value={extractedInfo.underwriting_agreement || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>IPO Grading</th>
                            <td>
                              <input
                                type="text"
                                id="ipo_grading"
                                value={extractedInfo.ipo_grading || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>


                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Exemption</th>
                            <td>
                              <input
                                type="text"
                                id="exemption"
                                value={extractedInfo.exemption || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Business</th>
                            <td>
                              <textarea
                                type="text"
                                id="business"
                                value={extractedInfo.business || ""}
                                className="form-control"
                                onChange={handleInputChange}
                                rows="4"
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>In Principle Approval</th>
                            <td>
                              <input
                                type="text"
                                id="in_principle_approval"
                                value={extractedInfo.in_principle_approval || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>About the Company</th>
                            <td>
                              <input
                                type="text"
                                id="about_company"
                                value={extractedInfo.about_company || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                          <tr>
                            <th style={{ backgroundColor: '#f8fafc' }}>Registered Office Address</th>
                            <td>
                              <input
                                type="text"
                                id="regd_office"
                                value={extractedInfo.regd_office || ""}
                                className="form-control"
                                onChange={handleInputChange}
                              />
                            </td>
                          </tr>
                        </tbody>
                      </table>
                      <button
                        className="btn btn-primary float-right col-4 mb-3"
                        onClick={handleUpdate}
                      >
                        Update
                      </button>
                      <button
                        className="btn btn-success btn-lg col-10 mt-5"
                        onClick={handleGenerateOfficeNote}
                      >
                        Generate Office Note
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        <ToastContainer />
      </div>
    </div>
  );
}

export default OfficeNote;
