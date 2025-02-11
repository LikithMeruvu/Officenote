// import React, { useState } from 'react';
// import axios from 'axios';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import ReactMarkdown from 'react-markdown';

// const News = () => {
//   const [companyName, setCompanyName] = useState('');
//   const [perplexityNews, setPerplexityNews] = useState([]);
//   const [gptNews, setGptNews] = useState([]);
//   const [combinedRedFlags, setCombinedRedFlags] = useState('');
//   const [expanded, setExpanded] = useState(false);
//   const [individualExpanded, setIndividualExpanded] = useState({});
//   const [activeTab, setActiveTab] = useState('perplexity');

//   const fetchNews = async () => {
//     try {
//       const formattedCompanyName = companyName.replace(/ /g, '%20');
//       console.log("Fetching news for company:", formattedCompanyName);
//       const response = await axios.get(
//         `http://localhost:8000/fetch_ipo_news?company_name=${formattedCompanyName}`
//       );

//       console.log("API Response:", response.data);

//       setPerplexityNews(response.data.perplexity_results || []);
//       setGptNews(response.data.gpt_results || []);
//       setCombinedRedFlags(response.data.combined_red_flags || '');
//       setExpanded(false);
//       setIndividualExpanded({});
//     } catch (error) {
//       console.error('Error fetching news:', error);
//       console.error('Error details:', error.response?.data);
//       setPerplexityNews([{
//         title: "Error",
//         result: `Failed to fetch news: ${error.message}`,
//         citations: []
//       }]);
//       setGptNews([]);
//       setCombinedRedFlags('');
//     }
//   };

//   const toggleExpandAll = () => {
//     setExpanded(!expanded);
//     if (!expanded) {
//       const expandedObj = {};
//       perplexityNews.forEach((item, index) => {
//         expandedObj[index] = true;
//       });
//       setIndividualExpanded(expandedObj);
//     } else {
//       setIndividualExpanded({});
//     }
//   };

//   const toggleIndividualExpand = (index) => {
//     const newExpanded = { ...individualExpanded };
//     newExpanded[index] = !newExpanded[index];
//     setIndividualExpanded(newExpanded);
//   };

//   return (
//     <div className="container mt-5">
//       <h1 className='title mb-5'>IPO Risk Analysis</h1>
//       <div className="mb-3 row">
//         <div className="col-md-8">
//           <input
//             type="text"
//             className="form-control"
//             placeholder="Enter company name"
//             value={companyName}
//             onChange={(e) => setCompanyName(e.target.value)}
//             onKeyDown={(e) => {
//               if (e.key === 'Enter') {
//                 setPerplexityNews([]);
//                 setGptNews([]);
//                 setCombinedRedFlags('');
//                 fetchNews();
//               }
//             }}
//           />
//         </div>
//         <div className="col-md-4 d-flex align-items-end">
//           <button 
//             className="btn btn-primary" 
//             onClick={() => {
//               setPerplexityNews([]);
//               setGptNews([]);
//               setCombinedRedFlags('');
//               fetchNews();
//             }}
//           >
//             Analyze Company
//           </button>
//         </div>
//       </div>

//       {combinedRedFlags && (
//         <div className="card mb-4 p-3 alert-danger">
//           <h3 className="mb-3">Critical Red Flags Summary</h3>
//           <ReactMarkdown>{combinedRedFlags}</ReactMarkdown>
//         </div>
//       )}

//       <ul className="nav nav-tabs mb-4">
//         <li className="nav-item">
//           <button
//             className={`nav-link ${activeTab === 'perplexity' ? 'active' : ''}`}
//             onClick={() => setActiveTab('perplexity')}
//           >
//             Perplexity Analysis ({perplexityNews.length})
//           </button>
//         </li>
//         <li className="nav-item">
//           <button
//             className={`nav-link ${activeTab === 'gpt' ? 'active' : ''}`}
//             onClick={() => setActiveTab('gpt')}
//           >
//             Deep Search Analysis ({gptNews.length})
//           </button>
//         </li>
//       </ul>

//       {activeTab === 'perplexity' && (
//         <div className="row">
//           <div className="col-md-12">
//             {perplexityNews.length > 0 && (
//               <button className="btn btn-secondary mb-3" onClick={toggleExpandAll}>
//                 {expanded ? 'Collapse All' : 'Expand All'}
//               </button>
//             )}
//             {perplexityNews.map((item, index) => (
//               <div key={index} className="mb-3">
//                 <div className="card p-3">
//                   <h5 
//                     onClick={() => toggleIndividualExpand(index)} 
//                     style={{ cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
//                   >
//                     <span>{item.title}</span>
//                     <span>{expanded || individualExpanded[index] ? '▼' : '▶'}</span>
//                   </h5>
//                   {(expanded || individualExpanded[index]) && (
//                     <div className="mt-3">
//                       <div className="mb-2">
//                         <strong>Query:</strong>
//                         <p>{item.query}</p>
//                       </div>
//                       <div className="mb-2">
//                         <strong>Analysis:</strong>
//                         <ReactMarkdown>{item.result}</ReactMarkdown>
//                       </div>
//                       {item.citations?.length > 0 && (
//                         <div>
//                           <strong>Sources:</strong>
//                           <ul>
//                             {item.citations.map((citation, i) => (
//                               <li key={i}>
//                                 <a 
//                                   href={citation.url} 
//                                   target="_blank" 
//                                   rel="noopener noreferrer"
//                                 >
//                                   {citation.title || citation.url}
//                                 </a>
//                               </li>
//                             ))}
//                           </ul>
//                         </div>
//                       )}
//                     </div>
//                   )}
//                 </div>
//               </div>
//             ))}
//           </div>
//         </div>
//       )}

//       {activeTab === 'gpt' && (
//         <div className="row">
//           <div className="col-md-12">
//             {gptNews.map((res, index) => (
//               <div key={index} className="mb-4">
//                 <div className="card p-3">
//                   <h5 className="mb-3">{res.question}</h5>
//                   <ReactMarkdown>{res.final_response}</ReactMarkdown>
//                   {res.citations?.length > 0 && (
//                     <div className="mt-3">
//                       <strong>References:</strong>
//                       <ul>
//                         {res.citations.map((citation, i) => (
//                           <li key={i}>
//                             <a
//                               href={citation}
//                               target="_blank"
//                               rel="noopener noreferrer"
//                             >
//                               {citation}
//                             </a>
//                           </li>
//                         ))}
//                       </ul>
//                     </div>
//                   )}
//                 </div>
//               </div>
//             ))}
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default News;


import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import ReactMarkdown from 'react-markdown';

const News = () => {
  const [companyName, setCompanyName] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('perplexity');
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    if (loading) {
      const messages = [
        'Analyzing company details...',
        'Searching regulatory databases...',
        'Checking financial records...',
        'Reviewing legal history...',
        'Compiling final report...'
      ];
      let current = 0;
      
      const interval = setInterval(() => {
        if (current < messages.length - 1) {
          current++;
          setProgress((current / messages.length) * 100);
          setStatusMessage(messages[current]);
        }
      }, 8000);

      return () => clearInterval(interval);
    }
  }, [loading]);

  const fetchNews = async () => {
    try {
      setLoading(true);
      setError('');
      setResults(null);
      setProgress(10);
      setStatusMessage('Initializing analysis...');

      const response = await axios.get(
        `http://localhost:8000/fetch_ipo_news?company_name=${encodeURIComponent(companyName)}`,
        { timeout: 3000000 } // 5 minute timeout
      );

      setResults({
        perplexity: response.data.perplexity_results || [],
        gpt: response.data.gpt_results || [],
        redFlags: response.data.combined_red_flags || 'No red flags found'
      });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
      setProgress(0);
      setStatusMessage('');
    }
  };

  const renderCitations = (citations) => {
    if (!citations?.length) return null;
    
    return (
      <div className="mt-2">
        <h6>Sources:</h6>
        <ul>
          {citations.map((cite, i) => (
            <li key={i}>
              <a href={cite.url || cite} target="_blank" rel="noopener noreferrer">
                {cite.title || cite.url || cite}
              </a>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="container mt-5">
      <h1 className="mb-4">IPO Risk Analysis Tool</h1>
      
      <div className="row mb-4">
        <div className="col-md-8">
          <div className="input-group">
            <input
              type="text"
              className="form-control"
              placeholder="Enter company name (e.g., Paytm, LIC)"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              disabled={loading}
            />
            <button 
              className="btn btn-primary"
              onClick={fetchNews}
              disabled={loading || !companyName.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner-border spinner-border-sm me-2" role="status" />
                  Analyzing...
                </>
              ) : 'Analyze'}
            </button>
          </div>
        </div>
      </div>

      {loading && (
        <div className="card mb-4">
          <div className="card-body text-center">
            <div className="progress mb-3">
              <div 
                className="progress-bar progress-bar-striped progress-bar-animated" 
                role="progressbar" 
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-muted mb-0">{statusMessage}</p>
            <small className="text-muted">This may take 2-3 minutes...</small>
          </div>
        </div>
      )}

      {error && (
        <div className="alert alert-danger">
          <strong>Error:</strong> {error}
        </div>
      )}

      {results && (
        <div>
          <div className="card mb-4 border-danger">
            <div className="card-header bg-danger text-white">
              <h5 className="mb-0">Red Flags Summary</h5>
            </div>
            <div className="card-body">
              <ReactMarkdown>{results.redFlags}</ReactMarkdown>
            </div>
          </div>

          <nav>
            <div className="nav nav-tabs" id="nav-tab">
              <button
                className={`nav-link ${activeTab === 'perplexity' ? 'active' : ''}`}
                onClick={() => setActiveTab('perplexity')}
              >
                Perplexity Analysis ({results.perplexity.length})
              </button>
              <button
                className={`nav-link ${activeTab === 'gpt' ? 'active' : ''}`}
                onClick={() => setActiveTab('gpt')}
              >
                google Search Analysis ({results.gpt.length})
              </button>
            </div>
          </nav>

          <div className="tab-content mt-4">
            {activeTab === 'perplexity' && (
              <div>
                {results.perplexity.map((item, index) => (
                  <div key={index} className="card mb-3">
                    <div className="card-body">
                      <h5 className="card-title">{item.title}</h5>
                      <div className="card-subtitle mb-2 text-muted">
                        <small>Search Query: {item.query}</small>
                      </div>
                      <ReactMarkdown className="card-text">{item.result}</ReactMarkdown>
                      {renderCitations(item.citations)}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === 'gpt' && (
              <div>
                {results.gpt.map((result, index) => (
                  <div key={index} className="card mb-3">
                    <div className="card-body">
                      <h5 className="card-title">{result.question}</h5>
                      <ReactMarkdown className="card-text">{result.final_response}</ReactMarkdown>
                      {renderCitations(result.citations)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default News;