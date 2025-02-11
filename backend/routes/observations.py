# import os
# import re
# import io
# import fitz  # PyMuPDF
# import pandas as pd
# from rapidfuzz import process, fuzz
# from markitdown import MarkItDown
# from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
# from fastapi.responses import JSONResponse
# from openai import OpenAI
# import tempfile
# import os
# from dotenv import load_dotenv
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# router = APIRouter(
#     prefix="/observations",
#     tags=["observations"]
# )

# ########################################
# #       Extract Table of Contents      #
# ########################################
# def extract_table_of_contents(pdf_stream):
#     """
#     Extracts the table of contents from the PDF and returns it as a DataFrame.
#     """
#     try:
#         with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
#             toc_start_page = None
#             for page_num in range(min(6, doc.page_count)):  # Check the first 5-6 pages
#                 page = doc.load_page(page_num)
#                 text = page.get_text("text")
#                 if "Contents" in text or "TABLE OF CONTENTS" in text:
#                     toc_start_page = page_num
#                     break

#             if toc_start_page is None:
#                 print("Table of Contents not found in the first 5-6 pages.")
#                 return None

#             toc_data = []
#             page_num = toc_start_page

#             while True:
#                 page = doc.load_page(page_num)
#                 links = [l for l in page.get_links() if l["kind"] in (fitz.LINK_GOTO, fitz.LINK_NAMED)]

#                 for link in links:
#                     rect = fitz.Rect(link['from'])
#                     link_text = page.get_text("text", clip=rect).strip()
#                     target_page = link.get("page") + 1 if link.get("page") is not None else None
#                     if link_text and target_page:
#                         toc_data.append({
#                             "Link Text": link_text,
#                             "Target Page": target_page
#                         })

#                 page_num += 1
#                 if page_num >= doc.page_count:
#                     break

#                 next_page_text = doc.load_page(page_num).get_text("text")
#                 if not any(keyword in next_page_text for keyword in ["SECTION", "....", "INTRODUCTION"]):
#                     break

#             if not toc_data:
#                 print("No TOC data extracted.")
#                 return None

#             df_links = pd.DataFrame(toc_data)

#             def clean_text(text):
#                 text = re.sub(r'\.{2,}.*', '', text)
#                 text = text.strip()
#                 return text

#             df_links['Link Text'] = df_links['Link Text'].apply(clean_text)
#             df_links['Type'] = df_links['Link Text'].apply(lambda x: 'Section' if 'SECTION' in x.upper() else 'Subject')

#             def remove_section_prefix(text):
#                 return re.sub(r'^SECTION\s*[IVXLC]+\s*[:\-]?\s*', '', text, flags=re.IGNORECASE).strip()

#             df_links['Cleaned Text'] = df_links['Link Text'].apply(remove_section_prefix)

#             entries = []
#             for idx, row in df_links.iterrows():
#                 entries.append({
#                     'Type': row['Type'],
#                     'Text': row['Link Text'],
#                     'CleanedText': row['Cleaned Text'],
#                     'StartingPage': row['Target Page']
#                 })

#             toc_entries = []
#             current_section = None
#             current_section_start = None
#             current_section_end = None

#             for idx, entry in enumerate(entries):
#                 if entry['Type'] == 'Section':
#                     if current_section is not None:
#                         current_section_end = entry['StartingPage'] - 1
#                         for e in toc_entries:
#                             if e['subject_section'] == current_section and e['ending_page_number'] is None:
#                                 e['ending_page_number'] = current_section_end
#                         for e in toc_entries:
#                             if e['subject_section'] == current_section and e['section_range'] is None:
#                                 e['section_range'] = f"{current_section_start}-{current_section_end}"

#                     current_section = entry['Text']
#                     current_section_start = entry['StartingPage']
#                     current_section_end = None
#                     toc_entries.append({
#                         'Type': entry['Type'],
#                         'subject': entry['Text'],
#                         'cleaned_subject': entry['CleanedText'],
#                         'starting_page_number': entry['StartingPage'],
#                         'ending_page_number': None,
#                         'subject_section': current_section,
#                         'section_range': None
#                     })
#                 else:
#                     if toc_entries and toc_entries[-1]['ending_page_number'] is None:
#                         previous_start = toc_entries[-1]['starting_page_number']
#                         current_start = entry['StartingPage']
#                         toc_entries[-1]['ending_page_number'] = max(previous_start, current_start - 1)
#                     toc_entries.append({
#                         'Type': entry['Type'],
#                         'subject': entry['Text'],
#                         'cleaned_subject': entry['CleanedText'],
#                         'starting_page_number': entry['StartingPage'],
#                         'ending_page_number': None,
#                         'subject_section': current_section,
#                         'section_range': None
#                     })

#             if current_section is not None:
#                 last_entry = toc_entries[-1]
#                 if last_entry['ending_page_number'] is None:
#                     last_entry['ending_page_number'] = doc.page_count
#                 current_section_end = last_entry['ending_page_number']

#                 for e in toc_entries:
#                     if e['subject_section'] == current_section and e['ending_page_number'] is None:
#                         e['ending_page_number'] = current_section_end

#                 for e in toc_entries:
#                     if e['subject_section'] == current_section and e['section_range'] is None:
#                         e['section_range'] = f"{current_section_start}-{current_section_end}"

#             toc_df = pd.DataFrame(toc_entries)
#             toc_df['subject_range'] = toc_df['starting_page_number'].astype(str) + " - " + toc_df['ending_page_number'].astype(str)
#             toc_df = toc_df[["Type", "subject", "cleaned_subject", "subject_range", "subject_section", "section_range", "starting_page_number", "ending_page_number"]]

#             return toc_df
#     except Exception as e:
#         print(f"Error extracting Table of Contents: {e}")
#         return None

# ########################################
# #    Extract Sub-PDFs and Convert to MD #
# ########################################
# def extract_sub_pdfs_and_convert(pdf_stream, toc_df, subcategories, output_dir):
#     """
#     Extracts sub-PDFs based on subcategories and converts them to Markdown files.

#     Parameters:
#     - pdf_stream: BytesIO object of the original PDF.
#     - toc_df: DataFrame containing the Table of Contents.
#     - subcategories: List of subcategories to extract.
#     - output_dir: Directory to save the extracted PDFs and MD files.

#     Returns:
#     - A dictionary mapping subcategories to their respective MD file paths.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     extracted_md_paths = {}

#     try:
#         with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
#             for subcat in subcategories:
#                 subcat_clean = subcat.lower().strip()
#                 subjects = toc_df['cleaned_subject'].tolist()
#                 subjects_lower = [s.lower() for s in subjects]

#                 best_match = process.extractOne(subcat_clean, subjects_lower, scorer=fuzz.partial_ratio)

#                 if best_match and best_match[1] > 60:  # Threshold can be adjusted
#                     matched_subject_clean, score, match_index = best_match
#                     matched_row = toc_df.iloc[match_index]
#                     start_page = int(matched_row['starting_page_number']) - 1
#                     end_page = int(matched_row['ending_page_number']) - 1

#                     if start_page < 0 or end_page >= doc.page_count or start_page > end_page:
#                         print(f"Invalid page range for '{subcat}'. Skipping.")
#                         continue

#                     sanitized_subcat = re.sub(r'\W+', '_', subcat.lower())
#                     sub_pdf_filename = f"temp_{sanitized_subcat}.pdf"
#                     sub_pdf_path = os.path.join(output_dir, sub_pdf_filename)

#                     new_doc = fitz.open()
#                     new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
#                     new_doc.save(sub_pdf_path)
#                     new_doc.close()
#                     print(f"Extracted '{subcat}' into '{sub_pdf_path}'")

#                     markitdown = MarkItDown()
#                     markdown_content = markitdown.convert(sub_pdf_path)
#                     md_filename = f"{sanitized_subcat}.md"
#                     md_path = os.path.join(output_dir, md_filename)

#                     with open(md_path, 'w', encoding='utf-8') as md_file:
#                         md_file.write(markdown_content.text_content)
#                     print(f"Converted '{subcat}' to Markdown at '{md_path}'")
#                     extracted_md_paths[subcat] = md_path
#                 else:
#                     print(f"No suitable match found for subcategory '{subcat}'. Skipping.")

#     except Exception as e:
#         print(f"Error extracting sub-PDFs and converting to Markdown: {e}")

#     return extracted_md_paths

# ########################################
# #           CSV Processing             #
# ########################################
# def get_unique_subcategories(csv_path):
#     """
#     Reads the CSV file and extracts unique subcategories.

#     Parameters:
#     - csv_path: Path to the CSV file.

#     Returns:
#     - A list of unique subcategories.
#     """
#     encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
#     for encoding in encodings_to_try:
#         try:
#             logger.info(f"Trying to read CSV with {encoding} encoding")
#             df = pd.read_csv(csv_path, encoding=encoding)
#             if 'sub_category' not in df.columns:
#                 logger.error(f"CSV file does not contain 'sub_category' column. Found columns: {df.columns.tolist()}")
#                 continue
#             subcategories = df['sub_category'].dropna().unique().tolist()
#             if subcategories:
#                 logger.info(f"Successfully read CSV with {encoding} encoding. Found {len(subcategories)} subcategories")
#                 return subcategories
#         except Exception as e:
#             logger.error(f"Failed to read CSV with {encoding} encoding: {str(e)}")
#             continue
    
#     logger.error("Failed to read CSV file with any encoding")
#     return []

# ########################################
# #           Markdown to Text Helper    #
# ########################################
# def get_text_from_markdown(md_path):
#     """
#     Reads the content from a Markdown file.

#     Parameters:
#     - md_path: Path to the Markdown file.

#     Returns:
#     - A string containing the extracted text.
#     """
#     try:
#         with open(md_path, 'r', encoding='utf-8') as f:
#             text = f.read()
#         return text.strip()
#     except Exception as e:
#         print(f"Error reading Markdown file '{md_path}': {e}")
#         return ""

# ########################################
# #              Ask ChatGPT             #
# ########################################
# def ask_chatgpt(messages, openai_api_key):
#     """
#     Sends messages to OpenAI's ChatCompletion API and returns the response.

#     Parameters:
#     - messages: A list of message dictionaries.
#     - openai_api_key: Your OpenAI API key.

#     Returns:
#     - The content of the AI's response.
#     """
#     client = OpenAI(api_key=openai_api_key)
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             temperature=0.1,
#         )
#         answer = response.choices[0].message.content.strip()
#         return answer
#     except Exception as e:
#         print(f"Error communicating with OpenAI: {e}")
#         return "Error: Could not retrieve answer."

# ########################################
# #              Q&A Processing          #
# ########################################
# def perform_qa(extracted_md_paths, questions_df, openai_api_key):
#     """
#     Performs Q&A using OpenAI's API based on the extracted Markdown content.

#     Parameters:
#     - extracted_md_paths: Dictionary mapping subcategories to their MD file paths.
#     - questions_df: DataFrame containing questions and their subcategories.
#     - openai_api_key: Your OpenAI API key.

#     Returns:
#     - A list of dictionaries containing questions, titles, and their answers.
#     """
#     qa_results = []

#     for idx, row in questions_df.iterrows():
#         question = row['question']
#         sub_category = row['sub_category']
#         title = row['title']  # Get the title from the DataFrame
#         print(f"\nProcessing Question {idx + 1}: '{question}' (Sub-category: '{sub_category}', Title: '{title}')")

#         if sub_category not in extracted_md_paths:
#             print(f"Sub-category '{sub_category}' not found in extracted Markdown files. Skipping.")
#             qa_results.append({
#                 'question': question,
#                 'title': title,  # Include title in output
#                 'sub_category': sub_category,
#                 'answer': "Sub-category not found in extracted content."
#             })
#             continue

#         md_path = extracted_md_paths[sub_category]
#         if not os.path.exists(md_path):
#             print(f"Markdown file '{md_path}' does not exist. Skipping.")
#             qa_results.append({
#                 'question': question,
#                 'title': title,  # Include title in output
#                 'sub_category': sub_category,
#                 'answer': "Markdown file not found."
#             })
#             continue

#         corpus_text = get_text_from_markdown(md_path)
#         if not corpus_text:
#             print(f"No text extracted from '{md_path}'. Skipping.")
#             qa_results.append({
#                 'question': question,
#                 'title': title,  # Include title in output
#                 'sub_category': sub_category,
#                 'answer': "No content found in Markdown file."
#             })
#             continue

#         messages = [
#             {
#                 "role": "system",
#                 "content": """
#                     You are an information extractor. The user will upload excerpts from a Draft Red Herring Prospectus filed by a company named JSW Cement Limited which intends to raise money through an Initial Public Offer. Please answer my questions based on this document only. Please be very accurate but only limited to this document. Temperature = 0. 
#                 """
#             },
#             {
#                 "role": "user",
#                 "content": f"Here is the content from the '{sub_category}' section of a Draft Red Herring Prospectus (DRHP):\n\n{corpus_text}"
#             },
#             {
#                 "role": "user",
#                 "content": f"Question: {question}"
#             }
#         ]

#         answer = ask_chatgpt(messages, openai_api_key)
#         if not answer:
#             answer = "No answer provided."

#         qa_results.append({
#             'question': question,
#             'title': title,  # Include title in output
#             'sub_category': sub_category,
#             'answer': answer
#         })

#     return qa_results

# ########################################
# #          Main Processing             #
# ########################################
# def process_pdf(pdf_file, csv_path, openai_api_key):
#     """
#     Main function to process the PDF and CSV files, perform Q&A, and return results.

#     Parameters:
#     - pdf_file: UploadFile object of the PDF.
#     - csv_path: Path to the CSV file.
#     - openai_api_key: Your OpenAI API key.

#     Returns:
#     - List of Q&A results.
#     """
#     try:
#         pdf_bytes = pdf_file.file.read()
#         pdf_stream = io.BytesIO(pdf_bytes)
#     except Exception as e:
#         logger.error(f"Error reading PDF file: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Error reading PDF file: {e}")

#     # Try reading CSV with different encodings
#     encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
#     questions_df = None
    
#     for encoding in encodings_to_try:
#         try:
#             logger.info(f"Trying to read CSV with {encoding} encoding")
#             questions_df = pd.read_csv(csv_path, encoding=encoding)
#             if all(col in questions_df.columns for col in ['question', 'sub_category', 'title']):
#                 logger.info(f"Successfully read CSV with {encoding} encoding")
#                 break
#         except Exception as e:
#             logger.error(f"Failed to read CSV with {encoding} encoding: {str(e)}")
#             continue
    
#     if questions_df is None:
#         raise HTTPException(status_code=400, detail="Failed to read CSV file with any encoding")

#     subcategories = questions_df['sub_category'].dropna().unique().tolist()
#     if not subcategories:
#         raise HTTPException(status_code=400, detail="No subcategories found in the CSV file.")

#     # Reset stream position if needed
#     pdf_stream.seek(0)
#     toc_df = extract_table_of_contents(pdf_stream)
#     if toc_df is None or toc_df.empty:
#         raise HTTPException(status_code=400, detail="Failed to extract Table of Contents from PDF.")

#     present_subcategories = []
#     toc_subcategories = toc_df['cleaned_subject'].unique().tolist()
#     for subcat in subcategories:
#         match = process.extractOne(subcat.lower(), [s.lower() for s in toc_subcategories], scorer=fuzz.partial_ratio)
#         if match and match[1] > 60:
#             present_subcategories.append(subcat)
#         else:
#             logger.warning(f"Subcategory '{subcat}' not found in PDF TOC.")

#     if not present_subcategories:
#         raise HTTPException(status_code=400, detail="No matching subcategories found in PDF TOC.")

#     extracted_md_paths = extract_sub_pdfs_and_convert(pdf_stream, toc_df, present_subcategories, output_dir=tempfile.gettempdir())

#     if not extracted_md_paths:
#         raise HTTPException(status_code=400, detail="No sections were successfully extracted and converted.")

#     qa_results = perform_qa(extracted_md_paths, questions_df, openai_api_key)

#     return qa_results

# ########################################
# #               API Endpoint           #
# ########################################

# @router.post("/process")
# async def process_file(pdf: UploadFile = File(...)):
#     """
#     Endpoint to process uploaded PDF file and return Q&A results as JSON.
#     """
#     logger.info(f"Starting to process file: {pdf.filename}")
    
#     try:
#         # Validate file type
#         if not (pdf.filename.endswith('.pdf') or pdf.filename.endswith('.PDF')):
#             logger.error(f"Invalid file type: {pdf.filename}")
#             raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")

#         # Get current directory and CSV path
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         csv_path = os.path.join(current_dir, "Observations_Prompts.csv")
#         print(f"Looking for CSV at: {csv_path}")
#         logger.info(f"Looking for CSV at: {csv_path}")
        
#         if not os.path.exists(csv_path):
#             logger.error(f"CSV file not found at: {csv_path}")
#             raise HTTPException(status_code=500, detail="Required CSV file not found")

#         # Create temp directory
#         temp_dir = os.path.join(current_dir, "temp")
#         if not os.path.exists(temp_dir):
#             os.makedirs(temp_dir)
#             logger.info(f"Created temp directory at: {temp_dir}")

#         # Process PDF
#         logger.info("Starting PDF processing...")
#         qa_results = process_pdf(pdf, csv_path, os.getenv("OPENAI_API_KEY"))
#         logger.info("PDF processing completed successfully")

#         # Clean up
#         logger.info("Cleaning up temporary files...")
#         for root, dirs, files in os.walk(temp_dir, topdown=False):
#             for name in files:
#                 os.remove(os.path.join(root, name))
#         logger.info("Cleanup completed")

#         return JSONResponse(content={
#             "qa_results": qa_results,
#             "status": "success"
#         })

#     except Exception as e:
#         logger.error(f"Error processing file: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))



import os
import re
import io
import fitz  # PyMuPDF
import pandas as pd
from rapidfuzz import process, fuzz
from markitdown import MarkItDown
from fastapi.responses import JSONResponse
from openai import OpenAI
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, APIRouter
from typing import List

app = FastAPI()  # Create the main FastAPI application instance
router = APIRouter(prefix="/observations") # Define APIRouter with a prefix

########################################
#       Extract Table of Contents      #
########################################
# ... (rest of your extract_table_of_contents function - no changes needed)
def extract_table_of_contents(pdf_stream):
    """
    Extracts the table of contents from the PDF and returns it as a DataFrame.
    """
    try:
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            toc_start_page = None
            for page_num in range(min(6, doc.page_count)):  # Check the first 5-6 pages
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if "Contents" in text or "TABLE OF CONTENTS" in text:
                    toc_start_page = page_num
                    break

            if toc_start_page is None:
                print("Table of Contents not found in the first 5-6 pages.")
                return None

            toc_data = []
            page_num = toc_start_page

            while True:
                page = doc.load_page(page_num)
                links = [l for l in page.get_links() if l["kind"] in (fitz.LINK_GOTO, fitz.LINK_NAMED)]

                for link in links:
                    rect = fitz.Rect(link['from'])
                    link_text = page.get_text("text", clip=rect).strip()
                    target_page = link.get("page") + 1 if link.get("page") is not None else None
                    if link_text and target_page:
                        toc_data.append({
                            "Link Text": link_text,
                            "Target Page": target_page
                        })

                page_num += 1
                if page_num >= doc.page_count:
                    break

                next_page_text = doc.load_page(page_num).get_text("text")
                if not any(keyword in next_page_text for keyword in ["SECTION", "....", "INTRODUCTION"]):
                    break

            if not toc_data:
                print("No TOC data extracted.")
                return None

            df_links = pd.DataFrame(toc_data)

            def clean_text(text):
                text = re.sub(r'\.{2,}.*', '', text)
                text = text.strip()
                return text

            df_links['Link Text'] = df_links['Link Text'].apply(clean_text)
            df_links['Type'] = df_links['Link Text'].apply(lambda x: 'Section' if 'SECTION' in x.upper() else 'Subject')

            def remove_section_prefix(text):
                return re.sub(r'^SECTION\s*[IVXLC]+\s*[:\-]?\s*', '', text, flags=re.IGNORECASE).strip()

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

            return toc_df
    except Exception as e:
        print(f"Error extracting Table of Contents: {e}")
        return None

########################################
#    Extract Sub-PDFs and Convert to MD #
########################################
# ... (rest of your extract_sub_pdfs_and_convert function - no changes needed)
def extract_sub_pdfs_and_convert(pdf_stream, toc_df, subcategories, output_dir):
    """
    Extracts sub-PDFs based on subcategories and converts them to Markdown files.

    Parameters:
    - pdf_stream: BytesIO object of the original PDF.
    - toc_df: DataFrame containing the Table of Contents.
    - subcategories: List of subcategories to extract.
    - output_dir: Directory to save the extracted PDFs and MD files.

    Returns:
    - A dictionary mapping subcategories to their respective MD file paths.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extracted_md_paths = {}

    try:
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            for subcat in subcategories:
                subcat_clean = subcat.lower().strip()
                subjects = toc_df['cleaned_subject'].tolist()
                subjects_lower = [s.lower() for s in subjects]

                best_match = process.extractOne(subcat_clean, subjects_lower, scorer=fuzz.partial_ratio)

                if best_match and best_match[1] > 60:  # Threshold can be adjusted
                    matched_subject_clean, score, match_index = best_match
                    matched_row = toc_df.iloc[match_index]
                    start_page = int(matched_row['starting_page_number']) - 1
                    end_page = int(matched_row['ending_page_number']) - 1

                    if start_page < 0 or end_page >= doc.page_count or start_page > end_page:
                        print(f"Invalid page range for '{subcat}'. Skipping.")
                        continue

                    sanitized_subcat = re.sub(r'\W+', '_', subcat.lower())
                    sub_pdf_filename = f"temp_{sanitized_subcat}.pdf"
                    sub_pdf_path = os.path.join(output_dir, sub_pdf_filename)

                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                    new_doc.save(sub_pdf_path)
                    new_doc.close()
                    print(f"Extracted '{subcat}' into '{sub_pdf_path}'")

                    markitdown = MarkItDown()
                    markdown_content = markitdown.convert(sub_pdf_path)
                    md_filename = f"{sanitized_subcat}.md"
                    md_path = os.path.join(output_dir, md_filename)

                    with open(md_path, 'w', encoding='utf-8') as md_file:
                        md_file.write(markdown_content.text_content)
                    print(f"Converted '{subcat}' to Markdown at '{md_path}'")
                    extracted_md_paths[subcat] = md_path
                else:
                    print(f"No suitable match found for subcategory '{subcat}'. Skipping.")

    except Exception as e:
        print(f"Error extracting sub-PDFs and converting to Markdown: {e}")

    return extracted_md_paths

########################################
#           CSV Processing             #
########################################
# ... (rest of your CSV processing function - no changes needed)
def get_unique_subcategories(csv_path):
    """
    Reads the CSV file and extracts unique subcategories.

    Parameters:
    - csv_path: Path to the CSV file.

    Returns:
    - A list of unique subcategories.
    """
    try:
        df_questions = pd.read_csv(csv_path, encoding='utf-8')
        subcategories = df_questions['sub_category'].dropna().unique().tolist()
        print(f"Unique subcategories extracted from CSV: {subcategories}")
        return subcategories
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

########################################
#           Markdown to Text Helper    #
########################################
# ... (rest of your Markdown to Text Helper function - no changes needed)
def get_text_from_markdown(md_path):
    """
    Reads the content from a Markdown file.

    Parameters:
    - md_path: Path to the Markdown file.

    Returns:
    - A string containing the extracted text.
    """
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text.strip()
    except Exception as e:
        print(f"Error reading Markdown file '{md_path}': {e}")
        return ""

########################################
#              Ask ChatGPT             #
########################################
# ... (rest of your Ask ChatGPT function - no changes needed)
def ask_chatgpt(messages, openai_api_key):
    """
    Sends messages to OpenAI's ChatCompletion API and returns the response.

    Parameters:
    - messages: A list of message dictionaries.
    - openai_api_key: Your OpenAI API key.

    Returns:
    - The content of the AI's response.
    """
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return "Error: Could not retrieve answer."

########################################
#              Q&A Processing          #
########################################
# ... (rest of your Q&A Processing function - no changes needed)
def perform_qa(extracted_md_paths, questions_df, openai_api_key):
    """
    Performs Q&A using OpenAI's API based on the extracted Markdown content.

    Parameters:
    - extracted_md_paths: Dictionary mapping subcategories to their MD file paths.
    - questions_df: DataFrame containing questions and their subcategories.
    - openai_api_key: Your OpenAI API key.

    Returns:
    - A list of dictionaries containing questions and their answers.
    """
    qa_results = []

    for idx, row in questions_df.iterrows():
        question = row['question']
        sub_category = row['sub_category']
        print(f"\nProcessing Question {idx + 1}: '{question}' (Sub-category: '{sub_category}')")

        if sub_category not in extracted_md_paths:
            print(f"Sub-category '{sub_category}' not found in extracted Markdown files. Skipping.")
            qa_results.append({
                'question': question,
                'sub_category': sub_category,
                'answer': "Sub-category not found in extracted content."
            })
            continue

        md_path = extracted_md_paths[sub_category]
        if not os.path.exists(md_path):
            print(f"Markdown file '{md_path}' does not exist. Skipping.")
            qa_results.append({
                'question': question,
                'sub_category': sub_category,
                'answer': "Markdown file not found."
            })
            continue

        corpus_text = get_text_from_markdown(md_path)
        if not corpus_text:
            print(f"No text extracted from '{md_path}'. Skipping.")
            qa_results.append({
                'question': question,
                'sub_category': sub_category,
                'answer': "No content found in Markdown file."
            })
            continue

        messages = [
            {
                "role": "system",
                "content": """
                    You are a knowledgeable assistant helping to answer questions based on provided documents.
                    Provide clear, concise, and accurate answers based on the content.
                """
            },
            {
                "role": "user",
                "content": f"Here is the content from the '{sub_category}' section of a Draft Red Herring Prospectus (DRHP):\n\n{corpus_text}"
            },
            {
                "role": "user",
                "content": f"Question: {question}"
            }
        ]

        answer = ask_chatgpt(messages, openai_api_key)
        if not answer:
            answer = "No answer provided."

        qa_results.append({
            'question': question,
            'sub_category': sub_category,
            'answer': answer
        })

    return qa_results

########################################
#          Main Processing             #
########################################
# ... (rest of your main processing function - no changes needed)
def process_pdf(pdf_file, csv_path, openai_api_key):
    """
    Main function to process the PDF and CSV files, perform Q&A, and return results.

    Parameters:
    - pdf_file: UploadFile object of the PDF.
    - csv_path: Path to the CSV file.
    - openai_api_key: Your OpenAI API key.

    Returns:
    - List of Q&A results.
    """
    try:
        pdf_bytes = pdf_file.file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF file: {e}")

    subcategories = get_unique_subcategories(csv_path)
    if not subcategories:
        raise HTTPException(status_code=400, detail="No subcategories found in the CSV file.")

    # Reset stream position if needed
    pdf_stream.seek(0)
    toc_df = extract_table_of_contents(pdf_stream)
    if toc_df is None or toc_df.empty:
        raise HTTPException(status_code=400, detail="Failed to extract Table of Contents from PDF.")

    present_subcategories = []
    toc_subcategories = toc_df['cleaned_subject'].unique().tolist()
    for subcat in subcategories:
        match = process.extractOne(subcat.lower(), [s.lower() for s in toc_subcategories], scorer=fuzz.partial_ratio)
        if match and match[1] > 60:
            present_subcategories.append(subcat)
        else:
            print(f"Subcategory '{subcat}' not found in PDF TOC.")

    if not present_subcategories:
        raise HTTPException(status_code=400, detail="No matching subcategories found in PDF TOC.")

    extracted_md_paths = extract_sub_pdfs_and_convert(pdf_stream, toc_df, present_subcategories, output_dir=tempfile.gettempdir())

    if not extracted_md_paths:
        raise HTTPException(status_code=400, detail="No sections were successfully extracted and converted.")

    pdf_stream.seek(0)
    try:
        questions_df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    qa_results = perform_qa(extracted_md_paths, questions_df, openai_api_key)

    return qa_results

########################################
#           API Endpoint           #
########################################
# ... (rest of your /process endpoint - no changes needed if you intend to use it)
@router.post("/process")
async def process_file(pdf: UploadFile = File(...)):
    """
    Endpoint to process uploaded PDF file and return Q&A results as JSON.
    The CSV file is read from a local relative path.
    """
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")

    # Define the relative path to the CSV file
    csv_relative_path = "Observations_Prompts.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_relative_path)

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail=f"CSV file '{csv_relative_path}' not found in the server.")

    # OpenAI API key from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

    try:
        qa_results = process_pdf(pdf, csv_path, openai_api_key)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    return JSONResponse(content={"qa_results": qa_results})




########################################
# ... (rest of your get_questions_and_categories_v2 function - no changes needed)
def get_questions_and_categories_v2(excel_df):
    """
    Reads the DataFrame (format: questions, sub_cat_1, sub_cat_2, range_c1, range_c2)
    and extracts questions, subcategories (up to 2), and page ranges.

    Handles cases where sub_cat or range is missing or 0 or -1.

    Parameters:
    - excel_df: DataFrame read from the Excel file.

    Returns:
    - A list of dictionaries, each containing:
        {
          'question': str,
          'subcategories': List[str],
          'page_ranges': List[str]
        }
    """
    questions_data = []
    for _, row in excel_df.iterrows():
        question = row['questions']
        subcategories = []
        page_ranges = []

        # Subcategory 1
        sub_cat_1 = row['sub_cat_1']
        range_c1 = row['range_c1']

        if pd.notna(sub_cat_1) and sub_cat_1:  # Check for notna and not empty string
            subcategories.append(str(sub_cat_1))
            page_ranges.append(str(range_c1) if pd.notna(range_c1) else None) # Keep None for handling later
        else:
            subcategories.append(None) # Append None to maintain order
            page_ranges.append(None)

        # Subcategory 2
        sub_cat_2 = row['sub_cat_2']
        range_c2 = row['range_c2']
        if pd.notna(sub_cat_2) and sub_cat_2: # Check for notna and not empty string
            subcategories.append(str(sub_cat_2))
            page_ranges.append(str(range_c2) if pd.notna(range_c2) else None) # Keep None for handling later
        else:
            subcategories.append(None) # Append None to maintain order
            page_ranges.append(None)

        questions_data.append({
            'question': question,
            'subcategories': [sc for sc in subcategories if sc is not None], # Filter out None subcategories
            'page_ranges': [pr for pr in page_ranges if pr is not None] # Filter out None page_ranges
        })
    print(f"Questions and categories extracted from Excel: {questions_data}")
    return questions_data

########################################
# ... (rest of your extract_and_convert_v2 function - no changes needed)
def extract_and_convert_v2(pdf_stream, toc_df, subcategory, page_range, output_dir):
    """
    Extracts pages from the subcategory based on the page_range.

    - If page_range is '-1', extracts the entire subcategory.
    - If page_range is a positive number, extracts up to 'page_range' pages from the start.
    - If page_range is '0' or None, returns None (skips extraction).

    Parameters:
    - pdf_stream: BytesIO object of the original PDF.
    - toc_df: DataFrame containing the Table of Contents.
    - subcategory: Subcategory to extract.
    - page_range: Page range as string ('-1', '0', positive number, or None).
    - output_dir: Directory to save the extracted PDF and MD file.

    Returns:
    - The path to the generated Markdown file, or None if skipped or error.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not subcategory or not subcategory.strip():
        return None

    page_range_int = None  # Initialize page_range_int
    if page_range is not None:
        try:
            page_range_int = int(page_range)
        except ValueError:
            print(f"Invalid page range value: '{page_range}' for subcategory '{subcategory}'. Treating as None.")
            return None

    if page_range_int == 0:
        print(f"Page range is 0 for subcategory '{subcategory}'. Skipping.")
        return None


    try:
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            subcat_clean = subcategory.lower().strip()
            subjects = toc_df['cleaned_subject'].tolist()
            subjects_lower = [s.lower() for s in subjects]

            best_match = process.extractOne(subcat_clean, subjects_lower, scorer=fuzz.partial_ratio)

            if best_match and best_match[1] > 60:  # Threshold can be adjusted
                matched_subject_clean, score, match_index = best_match
                matched_row = toc_df.iloc[match_index]

                # Full subcategory's range
                full_start = int(matched_row['starting_page_number']) - 1
                full_end = int(matched_row['ending_page_number']) - 1

                if page_range_int == -1:
                    # Extract the entire matched subcategory
                    start_page = full_start
                    end_page = full_end
                elif page_range_int is not None and page_range_int > 0:
                    # Extract up to 'page_range' pages from the start
                    offset = page_range_int
                    start_page = full_start
                    # End page is the minimum of (full_start + offset - 1) and the subcategory end
                    end_page = min(full_start + offset - 1, full_end)
                else:
                    print(f"No valid page range provided for '{subcategory}'. Skipping.")
                    return None


                # Validate pages
                if start_page < 0 or end_page >= doc.page_count or start_page > end_page:
                    print(f"Invalid page range for '{subcategory}'. Skipping.")
                    return None

                sanitized_subcat = re.sub(r'\W+', '_', subcategory.lower())
                sub_pdf_filename = f"temp_{sanitized_subcat}.pdf"
                sub_pdf_path = os.path.join(output_dir, sub_pdf_filename)

                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                new_doc.save(sub_pdf_path)
                new_doc.close()
                print(f"Extracted '{subcategory}' (pages {start_page + 1}-{end_page + 1}) into '{sub_pdf_path}'")

                markitdown = MarkItDown()
                markdown_content = markitdown.convert(sub_pdf_path)
                md_filename = f"{sanitized_subcat}.md"
                md_path = os.path.join(output_dir, md_filename)

                with open(md_path, 'w', encoding='utf-8') as md_file:
                    md_file.write(markdown_content.text_content)
                print(f"Converted '{subcategory}' to Markdown at '{md_path}'")
                return md_path
            else:
                print(f"No suitable match found for subcategory '{subcategory}'. Skipping.")
                return None

    except Exception as e:
        print(f"Error extracting sub-PDF and converting to Markdown: {e}")
        return None

########################################
# ... (rest of your perform_qa_v2 function - no changes needed)
def perform_qa_v2(pdf_stream, toc_df, questions_data, output_dir, openai_api_key, cache_rules=None, prev_observations=None):
    """
    Performs Q&A for questions with multiple subcategories and page ranges, including optional cache rules and prev observations.

    - If page_range == '-1', it extracts the full subcategory.
    - Otherwise, it extracts only 'page_range' pages from that subcategory's start.
    - Skips subcategories with page_range '0' or None.

    Parameters:
    - pdf_stream: BytesIO object of the original PDF.
    - toc_df: DataFrame containing the Table of Contents.
    - questions_data: List of dictionaries with 'question', 'subcategories', and 'page_ranges'.
    - output_dir: Directory to save extracted PDFs and MD files.
    - openai_api_key: Your OpenAI API key.

    Returns:
    - A list of dictionaries containing:
        {
          'question': str,
          'answer': str
        }
    """
    qa_results = []

    for q_data in questions_data:
        question = q_data['question']
        subcategories = q_data['subcategories']
        page_ranges = q_data['page_ranges']
        print(f"\nProcessing Question: '{question}'")

        combined_corpus_text = ""
        for subcategory, page_range in zip(subcategories, page_ranges):
            # Skip if subcategory or page_range is None or page_range is '0'
            if not subcategory or page_range is None or page_range == '0':
                print(f"Skipping subcategory '{subcategory}' due to missing or zero page range.")
                continue

            md_path = extract_and_convert_v2(pdf_stream, toc_df, subcategory, page_range, output_dir)
            if md_path:
                corpus_text = get_text_from_markdown(md_path)
                if corpus_text:
                    combined_corpus_text += f"\n\nContent from '{subcategory}' (Page Range: {page_range}):\n{corpus_text}"

        if not combined_corpus_text.strip():
            print(f"No content found for subcategories of question '{question}'. Skipping.")
            qa_results.append({
                'question': question,
                'answer': "No content found for the specified subcategories."
            })
            continue

        # Prepare the messages for ChatGPT
        messages = [
            {
                "role": "system",
                "content": """
                    You are a knowledgeable assistant helping to answer questions based on provided documents.
                    Provide clear, concise, and accurate answers based on the content.
                """
            },
            {
                "role": "user",
                "content": (
                    f"Here is the combined content from the relevant sections of a Draft Red Herring Prospectus (DRHP):\n\n"
                    f"{combined_corpus_text}"
                ),
            },
        ]

        if cache_rules: # Add cache rules if provided
            messages.append({
                "role": "user",
                "content": f"Cache Rules: {cache_rules}"
            })
        if prev_observations: # Add prev observations if provided
            messages.append({
                "role": "user",
                "content": f"Previous Observations: {prev_observations}"
            })


        messages.append({ # Finally add the question
            "role": "user",
            "content": f"Question: {question}"
        })


        answer = ask_chatgpt(messages, openai_api_key)
        if not answer:
            answer = "No answer provided."

        qa_results.append({
            'question': question,
            'answer': answer
        })

    return qa_results

########################################
#          Main Processing for V2      #
########################################
# ... (rest of your process_pdf_v2 function - no changes needed)
def process_pdf_v2(pdf_file, excel_df, openai_api_key, cache_rules=None, prev_observations=None):
    """
    Main function to process PDF and Excel v2, perform Q&A, with optional cache rules and prev observations.

    Parameters:
    - pdf_file: UploadFile object of the PDF.
    - excel_df: Pandas DataFrame from the uploaded Excel file (v2 format: questions, sub_cat_1, sub_cat_2, range_c1, range_c2).
    - openai_api_key: Your OpenAI API key.

    Returns:
    - List of Q&A results.
    """
    try:
        pdf_bytes = pdf_file.file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF file: {e}")

    # Get questions and categories from Excel DataFrame
    questions_data = get_questions_and_categories_v2(excel_df)
    if not questions_data:
        raise HTTPException(status_code=400, detail="No questions or categories found in the Excel file.")

    # Extract Table of Contents
    pdf_stream.seek(0)
    toc_df = extract_table_of_contents(pdf_stream)
    if toc_df is None or toc_df.empty:
        raise HTTPException(status_code=400, detail="Failed to extract Table of Contents from PDF.")

    # Perform Q&A, passing cache_rules and prev_observations
    qa_results = perform_qa_v2(pdf_stream, toc_df, questions_data, tempfile.gettempdir(), openai_api_key, cache_rules, prev_observations)

    return qa_results

########################################
#           API Endpoint for V2        #
########################################
#           API Endpoint for V2        #
########################################
# ... (rest of your /process_v2 endpoint - with router. instead of app.)
@router.post("/process_v2")
async def process_file_v2(
    pdf: UploadFile = File(...),
    excel_file: UploadFile = File(...),
    cache_rules: str = Form(None),  # Receive cache_rules as form data
    prev_observations: str = Form(None)  # Receive prev_observations as form data
):
    """
    Endpoint to process uploaded PDF and Excel files for Q&A, with optional cache rules and previous observations.
    """
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")
    if excel_file.content_type != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid XLSX file.")

    try:
        excel_content = await excel_file.read()
        excel_stream = io.BytesIO(excel_content)
        excel_df = pd.read_excel(excel_stream)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {e}")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

    qa_results = process_pdf_v2(pdf, excel_df, openai_api_key, cache_rules, prev_observations) # Pass new params
    return JSONResponse(content={"qa_results": qa_results})


@router.post("/process_v2")
async def process_file_v2(
    pdf: UploadFile = File(...),
    excel_file: UploadFile = File(...),
    cache_rules: str = Form(None),  # Optional cache rules
    prev_observations: str = Form(None)  # Optional previous observations
):
    """
    Endpoint to process uploaded PDF file and Excel file and return Q&A results as JSON (version 2).
    Handles questions with multiple subcategories and page ranges.
    """
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded PDF file is invalid.")
    if excel_file.content_type != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        raise HTTPException(status_code=400, detail="Uploaded XLSX file is invalid.")

    try:
        excel_content = await excel_file.read()
        excel_stream = io.BytesIO(excel_content)
        excel_df = pd.read_excel(excel_stream)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {e}")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is missing.")

    qa_results = process_pdf_v2(pdf, excel_df, openai_api_key, cache_rules, prev_observations)
    return JSONResponse(content={"qa_results": qa_results})


########################################
#          Main Processing for V2      #
########################################
# ... (rest of your process_pdf_v2 function - no changes needed)
def process_pdf_v2(pdf_file, excel_df, openai_api_key, cache_rules=None, prev_observations=None): # Add params here
    """
    Main function to process PDF and Excel v2, perform Q&A, with optional cache rules and prev observations.
    """
    try:
        pdf_bytes = pdf_file.file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF file: {e}")

    questions_data = get_questions_and_categories_v2(excel_df)
    if not questions_data:
        raise HTTPException(status_code=400, detail="No questions/categories in Excel file.")

    pdf_stream.seek(0)
    toc_df = extract_table_of_contents(pdf_stream)
    if not toc_df.notna().any().any(): # Check if DataFrame is empty or contains only NaNs
        raise HTTPException(status_code=400, detail="Failed to extract TOC from PDF.")


    qa_results = perform_qa_v2(pdf_stream, toc_df, questions_data, tempfile.gettempdir(), openai_api_key, cache_rules, prev_observations) # Pass to perform_qa_v2

    return qa_results


########################################
# ... (rest of your perform_qa_v2 function - no changes needed)
def perform_qa_v2(pdf_stream, toc_df, questions_data, output_dir, openai_api_key, cache_rules=None, prev_observations=None): # Add params here
    """
    Performs Q&A for questions with multiple subcategories and page ranges, including optional cache rules and prev observations.
    """
    qa_results = []

    for q_data in questions_data:
        question = q_data['question']
        subcategories = q_data['subcategories']
        page_ranges = q_data['page_ranges']
        print(f"\nProcessing Question: '{question}'")

        combined_corpus_text = ""
        for subcategory, page_range in zip(subcategories, page_ranges):
            if not subcategory or page_range is None or page_range == '0':
                print(f"Skipping subcategory '{subcategory}' due to missing/zero page range.")
                continue

            md_path = extract_and_convert_v2(pdf_stream, toc_df, subcategory, page_range, output_dir)
            if md_path:
                corpus_text = get_text_from_markdown(md_path)
                if corpus_text:
                    combined_corpus_text += f"\n\nContent from '{subcategory}' (Page Range: {page_range}):\n{corpus_text}"

        if not combined_corpus_text.strip():
            print(f"No content found for subcategories of question '{question}'. Skipping.")
            qa_results.append({
                'question': question,
                'answer': "No content found for the specified subcategories."
            })
            continue

        messages = [
            {
                "role": "system",
                "content": """
                    You are a knowledgeable assistant helping to answer questions based on provided documents.
                    Provide clear, concise, and accurate answers based on the content.
                """
            },
            {
                "role": "user",
                "content": (
                    f"Here is the combined content from the relevant sections of a Draft Red Herring Prospectus (DRHP):\n\n"
                    f"{combined_corpus_text}"
                ),
            },
        ]

        if cache_rules: # Add cache rules if provided
            messages.append({
                "role": "user",
                "content": f"Cache Rules: {cache_rules}"
            })
        if prev_observations: # Add prev observations if provided
            messages.append({
                "role": "user",
                "content": f"Previous Observations: {prev_observations}"
            })


        messages.append({ # Finally add the question
            "role": "user",
            "content": f"Question: {question}"
        })


        answer = ask_chatgpt(messages, openai_api_key)
        if not answer:
            answer = "No answer provided."

        qa_results.append({
            'question': question,
            'answer': answer
        })

    return qa_results

app.include_router(router)
