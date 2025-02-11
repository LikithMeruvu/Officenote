import os
import requests
import openai
import time
import logging
import random
from fastapi import FastAPI, Query, Body, UploadFile, Form, APIRouter
from typing import List
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pandas as pd
import re
from rapidfuzz import process, fuzz
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

router = APIRouter()

def extract_table_of_contents(pdf_path):
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
    toc_df = toc_df[["Type", "subject", "cleaned_subject", "subject_range", "subject_section", "section_range", "starting_page_number", "ending_page_number"]]

    doc.close()
    logging.info("TOC extraction completed.")
    return toc_df


def get_corpus_text(pdf_path):
    logging.info("Extracting corpus text from PDF sections...")
    df = extract_table_of_contents(pdf_path)
    if df is None or df.empty:
        logging.info("No corpus text extracted (no TOC found).")
        return ""

    sections_to_extract = [
        "SECTION II: RISK FACTORS",
        "OUTSTANDING LITIGATION AND MATERIAL DEVELOPMENTS",
        "SECTION VII: OUR GROUP COMPANIES"
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
                corpus_text += f"\n=== {matched_row['subject']} ===\n" + section_text.strip() + "\n"
            except Exception as e:
                logging.error(f"Error extracting text for section '{section_name}': {e}")
    doc.close()
    logging.info("Corpus text extraction completed.")
    return corpus_text.strip()


def retry_with_exponential_backoff(func):
    """
    Decorator to retry a function call with exponential backoff and jitter when RateLimitError is encountered.
    """
    def wrapper(*args, **kwargs):
        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == max_retries:
                    logging.error("Max retries reached. Raising RateLimitError.")
                    raise
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"Rate limit hit. Waiting {delay:.2f} seconds before retrying (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
    return wrapper


@retry_with_exponential_backoff
def call_openai_api(model, messages):
    try:
        client = OpenAI(api_key='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')
        response = client.chat.completions.create(
            model=model,  # Use the correct model name
            messages=messages,
            temperature=0,
            response_format= { type: "json_object" }
        )
        answer = response.choices[0].message['content']
        return answer
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return json.dumps({"error": "OpenAI API error"})
    logging.info("Received response from OpenAI API.")
    return response


def answer_multiple_questions(corpus_text: str, questions: List[str]) -> List[str]:
    """
    Ask all questions at once in a single OpenAI call.
    """
    prompt = f"{corpus_text}\n\nPlease answer the following questions:\n"
    for i, q in enumerate(questions, start=1):
        prompt += f"{i}. {q}\n"
    prompt += "\nAnswer each question in order, prefixing each answer with the question number.\n"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    response = call_openai_api("chatgpt-4o-latest", messages)
    answer = response.choices[0].message['content'].strip()
    lines = answer.split('\n')
    answers = []
    for i in range(1, len(questions)+1):
        prefix = f"{i}."
        line_answer = None
        for line in lines:
            if line.strip().startswith(prefix):
                line_answer = line.strip()[len(prefix):].strip()
                break
        if line_answer is None:
            line_answer = "No answer provided."
        answers.append(line_answer)
    return answers


@router.post("/process")
async def process_endpoint(
    company_name: str = Form(...),
    drhp_pdf: UploadFile = Form(...),
    promoters_csv: UploadFile = Form(...)
    ):
    """
    Endpoint to process DRHP and PDF to extract corpus and run OpenAI queries for each promoter.
    """
    logging.info("Processing request for /process...")
    pdf_content = await drhp_pdf.read()
    pdf_path = "temp_drhp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    logging.info("PDF saved locally.")

    csv_content = await promoters_csv.read()
    df_promoters = pd.read_csv(io.BytesIO(csv_content))
    logging.info(f"Loaded CSV with {len(df_promoters)} promoters.")

    corpus = get_corpus_text(pdf_path)

    questions = [
        "What are the main risk factors mentioned for {person_name}, who serves as a {person_role} at {company_name}?",
        "Is there any litigation information related to {person_name}, a {person_role} at {company_name}?",
        "Based on {company_name}'s DRHP, what are the key legal issues associated with {person_name}, who serves as a {person_role}?"
    ]

    results = []
    total_promoters = len(df_promoters)
    for idx, row in df_promoters.iterrows():
        person_name = row["NAME"]
        person_role = row["Designation"]
        logging.info(f"Processing promoter {idx+1}/{total_promoters}: {person_name} ({person_role})")

        formatted_questions = [
            q.format(company_name=company_name, person_name=person_name, person_role=person_role)
            for q in questions
        ]

        answers = answer_multiple_questions(corpus, formatted_questions)
        result_obj = {
            "name": person_name,
            "role": person_role,
            "company_name": company_name,
            "answers": [
                {"question": formatted_questions[i], "response": answers[i]} 
                for i in range(len(questions))
            ]
        }
        results.append(result_obj)

    logging.info("All promoters processed. Returning results.")
    return results


# For Perplexity endpoint
QUESTION_TEMPLATES = [
    "Find and summarize the latest news articles about the upcoming IPO of {company_name} in India. Focus on any potential controversies or legal scrutiny involving {person_name}, a {person_role}, or any directors.",
    "Are there any interviews given by {person_name}, who serves as a {person_role} at {company_name}, regarding the IPO? If so, summarize their key statements.",
    "Search for legal troubles or lawsuits involving {person_name}, a {person_role} at {company_name}, that could impact the company's reputation."
]

def call_perplexity_api(query: str):
    """
    Call the Perplexity API for a single query.
    """
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
        data = response.json()
        result = data.get("choices", [{}])[0].get("message", {}).get("content", "No news available.")
        citations = data.get("citations", [])
        return result, citations
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}", []


@router.post("/process_perplexity")
async def process_perplexity(
    company_name: str = Form(...),
    promoters_csv: UploadFile = Form(...)
    ):
    """
    Endpoint that:
    - Reads a CSV of promoters (with columns NAME, Designation)
    - For each promoter, runs 3 queries on Perplexity individually
    - Returns JSON grouped by promoter, each with their 3 queries and results
    """
    logging.info("Processing request for /process_perplexity...")

    csv_content = await promoters_csv.read()
    df_promoters = pd.read_csv(io.BytesIO(csv_content))
    logging.info(f"Loaded CSV with {len(df_promoters)} promoters.")

    final_results = []

    # Iterate over each promoter
    for idx, row in df_promoters.iterrows():
        person_name = row["NAME"]
        person_role = row["Designation"]
        logging.info(f"Processing promoter {idx+1}/{len(df_promoters)}: {person_name} ({person_role})")

        promoter_queries = []

        # Generate and call 3 queries for this person
        for template in QUESTION_TEMPLATES:
            query = template.format(company_name=company_name, person_name=person_name, person_role=person_role)
            # Call perplexity for each query
            result, citations = call_perplexity_api(query)
            promoter_queries.append({
                "query": query,
                "result": result,
                "citations": citations
            })

        # Add this promoter's data to the final results
        final_results.append({
            "name": person_name,
            "role": person_role,
            "queries": promoter_queries
        })

        # Optional: If needed, delay between promoters
        # time.sleep(1)

    # Final response
    return {
        "company_name": company_name,
        "results": final_results
    }
