import React, { useState } from 'react';
import { Accordion, Card } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../App.css';

const Home = () => {
  return (
    <div className="container-fluid">
      <div className="page mx-auto pt-4">
        <h1 className="title">Processing Draft Red Herring Prospectuses (DRHPs)</h1>
        
        <div className="card shadow-sm mt-4">
          <div className="card-body">
            <p className="mb-3 text-start">
              Draft Red Herring Prospectuses (<strong>DRHPs</strong> hereinafter) are preliminary offering documents submitted by companies intending to raise funds through an Initial Public Offering (IPO) or Follow-on Public Offering (FPO). The DRHP is scrutinized to ensure compliance with the SEBI (Issue of Capital and Disclosure Requirements) Regulations, 2018 (<strong>ICDR Regulations</strong>).
            </p>
            <p className="mb-3 text-start">
              The examination of DRHPs involves checking for compliance with regulatory requirements, ensuring accurate disclosures, analyzing potential risks, and ensuring that all necessary details are adequately covered.
            </p>
          </div>
        </div>

        <div className="mt-4">
          <Accordion>
            <Accordion.Item eventKey="1" className="mb-3 shadow-sm">
              <Accordion.Header>
                <strong>Observations</strong>
              </Accordion.Header>
              <Accordion.Body>
                <div className="card border-0">
                  <div className="card-body">
                    <h2 className="title">Observations Module</h2>
                    <p className="text-start">
                      This module compares the DRHP against past observations made by SEBI on similar filings. It checks for adherence to previously raised concerns or requirements and ensures that these have been addressed.
                    </p>
                    <p className="text-start">
                      It also provides a summary of recurring issues across similar DRHPs, helping the Dealing Officer proactively identify areas of concern.
                    </p>
                  </div>
                </div>
              </Accordion.Body>
            </Accordion.Item>

            <Accordion.Item eventKey="2" className="mb-3 shadow-sm">
              <Accordion.Header>
                <strong>News</strong>
              </Accordion.Header>
              <Accordion.Body>
                <div className="card border-0">
                  <div className="card-body">
                    <h2 className="title">News Module</h2>
                    <p className="text-start">
                      This module leverages advanced language models (LLMs) to analyze news articles related to the company, its promoters, and directors. It identifies key themes, sentiments, and any adverse media coverage that may impact investor confidence or the company's credibility.
                    </p>
                    <p className="text-start">
                      The analysis is presented in an easy-to-read format, enabling the Dealing Officer to assess the public perception of the company and its stakeholders.
                    </p>
                  </div>
                </div>
              </Accordion.Body>
            </Accordion.Item>

            <Accordion.Item eventKey="3" className="mb-3 shadow-sm">
              <Accordion.Header>
                <strong>What's Missing?</strong>
              </Accordion.Header>
              <Accordion.Body>
                <div className="card border-0">
                  <div className="card-body">
                    <h2 className="title">What's Missing? Module</h2>
                    <p className="text-start">
                      This module checks whether all relevant information present in public sources, such as websites, news, and corporate disclosures, has been adequately disclosed in the DRHP. It identifies discrepancies or missing details and flags them for the Dealing Officer's review.
                    </p>
                    <p className="text-start">
                      The module ensures that the DRHP provides a comprehensive and transparent view of the company's operations, mitigating the risk of incomplete disclosures.
                    </p>
                  </div>
                </div>
              </Accordion.Body>
            </Accordion.Item>

            <Accordion.Item eventKey="5" className="mb-3 shadow-sm">
              <Accordion.Header>
                <strong>Database Check</strong>
              </Accordion.Header>
              <Accordion.Body>
                <div className="card border-0">
                  <div className="card-body">
                    <h2 className="title">Database Check Module</h2>
                    <p className="text-start">
                      This module lets a user upload a list of names - the company, its promoters, directors and key managerial persons. 
                      It then checks the names against the list of Prosecution cases available at SEBI Website <a href='https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes&cid=14' target='_blank' rel='noopener noreferrer'>(Click Here)</a> or the list or the Vanishing companies.
                    </p>
                  </div>
                </div>
              </Accordion.Body>
            </Accordion.Item>

            <Accordion.Item eventKey="6" className="mb-3 shadow-sm">
              <Accordion.Header>
                <strong>Office Note</strong>
              </Accordion.Header>
              <Accordion.Body>
                <div className="card border-0">
                  <div className="card-body">
                    <h2 className="title">Office Note Module</h2>
                    <p className="text-start">
                      This module facilitates the generation of Office Notes for processing DRHPs. The module extracts relevant information from the DRHP and organizes it into a structured format for internal review. It ensures that all essential sections are covered, highlights key details, and summarizes compliance findings.
                    </p>
                    <p className="text-start">
                      The generated Office Note is downloadable, providing the Dealing Officer with a ready-to-use document for further processing.
                    </p>
                  </div>
                </div>
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </div>
      </div>
    </div>
  );
};

export default Home;
