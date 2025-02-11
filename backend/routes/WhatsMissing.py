



import os
import requests
import time
import logging
import random
from fastapi import FastAPI, Query, Body, UploadFile, Form, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io
import json


# ========= LATEST OPENAI USAGE (v1.x) =========
from openai import OpenAI

from pydantic import BaseModel

# ----------------------------------------------------
# Logging + environment setup
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
load_dotenv()

router = APIRouter()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Latest OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------
# PDF / TOC Extraction
# ----------------------------------------------------
def extract_table_of_contents(pdf_path: str) -> Optional[pd.DataFrame]:
    logging.info("Extracting table of contents from PDF...")
    doc = fitz.open(pdf_path)

    toc_start_page = None
    for page_num in range(6):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if "Contents" in text or "TABLE OF CONTENTS" in text:
                toc_start_page = page_num
                break
        except Exception:
            continue

    if toc_start_page is None:
        doc.close()
        logging.info("No Table of Contents found.")
        return None

    toc_data = []
    page_num = toc_start_page

    while True:
        try:
            page = doc.load_page(page_num)
            links = [l for l in page.get_links() if l["kind"] in (fitz.LINK_GOTO, fitz.LINK_NAMED)]
        except Exception:
            break

        for link in links:
            try:
                rect = fitz.Rect(link['from'])
                link_text = page.get_text("text", clip=rect).strip()
                target_page = link.get("page") + 1
                if link_text:
                    toc_data.append({
                        "Link Text": link_text,
                        "Target Page": target_page
                    })
            except Exception:
                continue

        page_num += 1
        if page_num >= doc.page_count:
            break

        try:
            next_page_text = doc.load_page(page_num).get_text("text")
        except Exception:
            break

        # A simple heuristic for continuing searching
        if not any(keyword in next_page_text for keyword in ["SECTION", "....", "INTRODUCTION"]):
            break

    if not toc_data:
        doc.close()
        logging.info("No TOC data extracted.")
        return None

    df_links = pd.DataFrame(toc_data)

    def clean_text(text):
        text = re.sub(r'\.{2,}.*', '', text)
        return text.strip()

    df_links['Link Text'] = df_links['Link Text'].apply(clean_text)
    df_links['Type'] = df_links['Link Text'].apply(lambda x: 'Section' if 'SECTION' in x.upper() else 'Subject')

    def remove_section_prefix(text):
        return re.sub(r'^SECTION\s*[IVXLC]+\s*:\s*', '', text, flags=re.IGNORECASE).strip()

    df_links['Cleaned Text'] = df_links['Link Text'].apply(remove_section_prefix)

    entries = []
    for idx, row in df_links.iterrows():
        entries.append({
            'Type': row['Type'],
            'Text': row['Link Text'],
            'CleanedText': row['Cleaned Text'],
            'StartingPage': row['Target Page']
        })

    toc_entries = []
    current_section = None
    current_section_start = None
    current_section_end = None

    for idx, entry in enumerate(entries):
        if entry['Type'] == 'Section':
            if current_section is not None:
                current_section_end = entry['StartingPage'] - 1
                for e in toc_entries:
                    if e['subject_section'] == current_section and e['ending_page_number'] is None:
                        e['ending_page_number'] = current_section_end
                for e in toc_entries:
                    if e['subject_section'] == current_section and e['section_range'] is None:
                        e['section_range'] = f"{current_section_start}-{current_section_end}"
            current_section = entry['Text']
            current_section_start = entry['StartingPage']
            current_section_end = None
            toc_entries.append({
                'Type': entry['Type'],
                'subject': entry['Text'],
                'cleaned_subject': entry['CleanedText'],
                'starting_page_number': entry['StartingPage'],
                'ending_page_number': None,
                'subject_section': current_section,
                'section_range': None
            })
        else:
            if toc_entries and toc_entries[-1]['ending_page_number'] is None:
                previous_start = toc_entries[-1]['starting_page_number']
                current_start = entry['StartingPage']
                toc_entries[-1]['ending_page_number'] = max(previous_start, current_start - 1)
            toc_entries.append({
                'Type': entry['Type'],
                'subject': entry['Text'],
                'cleaned_subject': entry['CleanedText'],
                'starting_page_number': entry['StartingPage'],
                'ending_page_number': None,
                'subject_section': current_section,
                'section_range': None
            })

    if current_section is not None:
        last_entry = toc_entries[-1]
        if last_entry['ending_page_number'] is None:
            last_entry['ending_page_number'] = doc.page_count
        current_section_end = last_entry['ending_page_number']
        for e in toc_entries:
            if e['subject_section'] == current_section and e['ending_page_number'] is None:
                e['ending_page_number'] = current_section_end
        for e in toc_entries:
            if e['subject_section'] == current_section and e['section_range'] is None:
                e['section_range'] = f"{current_section_start}-{current_section_end}"

    toc_df = pd.DataFrame(toc_entries)
    toc_df['subject_range'] = toc_df['starting_page_number'].astype(str) + " - " + toc_df['ending_page_number'].astype(str)
    toc_df = toc_df[[
        "Type", "subject", "cleaned_subject", "subject_range",
        "subject_section", "section_range", "starting_page_number",
        "ending_page_number"
    ]]

    doc.close()
    logging.info("TOC extraction completed.")
    return toc_df


def get_corpus_text(pdf_path: str) -> str:
    logging.info("Extracting corpus text from PDF sections...")
    df = extract_table_of_contents(pdf_path)
    if df is None or df.empty:
        logging.info("No corpus text extracted (no TOC found).")
        return ""

    sections_to_extract = [
        "OUTSTANDING LITIGATION AND MATERIAL DEVELOPMENTS",
    ]

    corpus_text = ""
    doc = fitz.open(pdf_path)

    for section_name in sections_to_extract:
        df_sections = df[df['Type'] == 'Section']
        subjects = df_sections['subject'].tolist()
        subjects_lower = [s.lower() for s in subjects]
        section_name_clean = section_name.lower().strip()

        best_match = process.extractOne(section_name_clean, subjects_lower, scorer=fuzz.partial_ratio)
        if best_match:
            matched_subject_clean, score, match_index = best_match
            matched_row = df_sections.iloc[match_index]
            try:
                start_str, end_str = matched_row['section_range'].split('-')
                start_page = int(start_str) - 1
                end_page = int(end_str) - 1
                section_text = ""
                for p in range(start_page, end_page + 1):
                    page = doc.load_page(p)
                    section_text += page.get_text("text") + "\n"
                corpus_text += f"\n## {matched_row['subject']}\n{section_text.strip()}\n"
            except Exception as e:
                logging.error(f"Error extracting text for section '{section_name}': {e}")

    doc.close()
    logging.info("Corpus text extraction completed.")
    return corpus_text.strip()


# ----------------------------------------------------
# Latest OpenAI Chat Completion call (for v1.x)
# ----------------------------------------------------
def call_openai_chat_completion(prompt: str) -> Any:
    """
    Using openai==1.x style:
       client.chat.completions.create(model=..., messages=...)
    """
    logging.info("Sending request to OpenAI (latest library) ...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the correct chat model
            messages=[
                {"role": "system", "content": "You are an expert legal analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        logging.info("Received response from OpenAI chat completion endpoint.")
        return response
    except Exception as e:
        logging.error(f"Error calling OpenAI chat completion endpoint: {e}")
        return None


# ----------------------------------------------------
# Perplexity Calls
# ----------------------------------------------------
QUESTION_TEMPLATES = [
    "Find and summarize the latest news articles about the upcoming IPO of {company_name} in India. Focus on any potential controversies or legal scrutiny involving {person_name}, a {person_role}, or any directors.",
    "Are there any interviews given by {person_name}, who serves as a {person_role} at {company_name}, regarding the IPO? If so, summarize their key statements.",
    "Search for legal troubles or lawsuits involving {person_name}, a {person_role} at {company_name}, that could impact the company's reputation."
]


def call_perplexity_api(query: str):
    """
    Call the Perplexity API for a single query.
    """
    if not PERPLEXITY_API_KEY:
        logging.warning("PERPLEXITY_API_KEY is not set. Returning None.")
        return None

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "This is for a research report. Be accurate and detailed."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": 4000,
        "temperature": 0,
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Perplexity API: {e}")
        return None


# ----------------------------------------------------
# Analyze litigations with latest OpenAI
# ----------------------------------------------------
def analyze_litigation_with_openai(combined_content: str, person_name: str, company_name: str) -> str:
    """
    Analyzes the combined Perplexity response with OpenAI to extract litigation details.
    Returns a plain text response describing the litigations.
    If no litigations found, returns a message indicating no litigations were found.
    """
    if not combined_content:
        return "No content provided to analyze."

    prompt = f"""Analyze the following text and extract any information about litigations, lawsuits, or legal troubles related to {person_name} at {company_name}.
Provide a detailed description of the litigations found in the text. If no litigations are found, simply state that no litigations were found.

Text:
{combined_content}
"""

    openai_response = call_openai_chat_completion(prompt)
    if openai_response and openai_response.choices:
        return openai_response.choices[0].message.content.strip()
    else:
        return "Could not get response from analysis."


# ----------------------------------------------------
# Models, Endpoints
# ----------------------------------------------------
class LitigationInfo(BaseModel):
    litigation: str
    category: str


@router.post("/process_perplexity")
async def process_perplexity_endpoint(
    company_name: str = Form(...),
    person_name: str = Form(...),
    role: str = Form(...)
) -> Dict[str, Any]:
    """
    Endpoint to process Perplexity queries for a single person and company.
    1. Calls Perplexity with three different queries about the given company/person.
    2. Passes combined Perplexity text to latest OpenAI to see if there's any litigation info.
    3. Returns:
       - perplexity_responses: All raw Perplexity responses
       - found_litigations: Plain text response from OpenAI
    """
    logging.info(f"Processing request for /process_perplexity for {person_name} at {company_name} ({role})")

    all_perplexity_responses = []
    for template in QUESTION_TEMPLATES:
        query = template.format(company_name=company_name, person_name=person_name, person_role=role)
        perplexity_response = call_perplexity_api(query)
        if perplexity_response:
            all_perplexity_responses.append({
                "query": query,
                "response": perplexity_response
            })

    # Combine the content from all perplexity responses
    combined_content = ""
    for resp_data in all_perplexity_responses:
        content = (
            resp_data['response']
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        combined_content += f"\n\n{resp_data['query']}:\n{content}"

    # Attempt to find litigations using latest OpenAI
    found_litigations = analyze_litigation_with_openai(
        combined_content,
        person_name,
        company_name
    )

    return {
        "perplexity_responses": all_perplexity_responses,
        "found_litigations": found_litigations
    }



@router.post("/process")
async def process_endpoint(
    drhp_pdf: UploadFile = File(...),
    litigations: str = Form(...)
) -> List[Dict[str, Any]]:
    """
    Endpoint to process DRHP and verify litigations.
    1. Upload a PDF file ("drhp_pdf") via multipart/form-data.
    2. Provide a "litigations" string describing a suspected litigation or question.
    3. The service extracts key sections from the DRHP (Risk Factors, etc.).
    4. Sends the extracted text + your "litigations" statement to latest OpenAI for analysis.
    5. Returns a JSON list with the verification response.
    """
    logging.info("Processing request for /process...")

    # Save the PDF locally
    pdf_bytes = await drhp_pdf.read()
    pdf_path = "temp_drhp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    logging.info("PDF saved locally.")

    # Extract relevant text
    corpus_text = get_corpus_text(pdf_path)

    prompt = f"""Based on the following DRHP text, verify the truthfulness of the following statement regarding potential litigations: '{litigations}'.

Provide a detailed explanation of whether this statement is supported by the text and include any relevant details found within the DRHP.

DRHP Text:
{corpus_text}
"""

    openai_response = call_openai_chat_completion(prompt)
    if openai_response and openai_response.choices:
        response_content = openai_response.choices[0].message.content.strip()
    else:
        response_content = "Could not get response from analysis."

    # Clean up
    try:
        os.remove(pdf_path)
    except:
        pass

    return [{
        "provided_litigation_statement": litigations,
        "verification_response": response_content
    }]
