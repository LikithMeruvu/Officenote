import os
import re
import json
import base64
import shutil
import fitz  # PyMuPDF
import pandas as pd
from rapidfuzz import process, fuzz
from openai import OpenAI
import requests
from fastapi import FastAPI, File, UploadFile, Query, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from fastapi.responses import FileResponse
from docxtpl import DocxTemplate
import shutil
import os
import logging

# Define the temporary directory
TEMP_DIR = "temp"

router = APIRouter()
client = OpenAI(api_key="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")

def first_page_check(pdf_path):

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    text = page.get_text("text")
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if "6(1)" in text:
        reg_6 = "Regulation 6(1)"
    elif "6(2)" in text:
        reg_6 = "Regulation 6(2)"
    else:
        reg_6 = "Not found"

    if "equity shares" in text.lower():
        instrument = "Equity Shares"
    elif "convertible shares" in text.lower():
        instrument = "Convertible Shares"
    else:
        instrument = "Not found"
    
    if "book built" in text.lower():
        bidding_type = "Book Built Issue"
    elif "fixed price" in text.lower():
        bidding_type = "Fixed Price Issue"
    else:
        bidding_type = "Not found"

    doc.close()

    return reg_6, instrument, bidding_type

def abbreviate_name(name):
    words = name.split()
    abbreviation = "".join([word[0] for word in words])
    return abbreviation

def ask_chatgpt(messages):
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the correct model name
            messages=messages,
            response_format={"type": "json_object"}
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return json.dumps({"error": "OpenAI API error"})

########################################
#       Extract Table of Contents      #
########################################
def extract_table_of_contents(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return None
    
    toc_start_page = None
    for page_num in range(min(6, doc.page_count)):  # Check the first 5-6 pages
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if "Contents" in text or "TABLE OF CONTENTS" in text:
                toc_start_page = page_num
                break
        except Exception as e:
            print(f"Error loading page {page_num} in {pdf_path}: {e}")
            continue

    if toc_start_page is None:
        print(f"Table of Contents not found in the first 5-6 pages of {pdf_path}.")
        doc.close()
        return None

    toc_data = []
    page_num = toc_start_page

    while True:
        try:
            page = doc.load_page(page_num)
            links = [l for l in page.get_links() if l["kind"] in (fitz.LINK_GOTO, fitz.LINK_NAMED)]
        except Exception as e:
            print(f"Error loading page {page_num} in {pdf_path}: {e}")
            break

        for link in links:
            try:
                rect = fitz.Rect(link['from'])
                link_text = page.get_text("text", clip=rect).strip()
                target_page = link.get("page") + 1 if link.get("page") is not None else None
                if link_text and target_page:
                    toc_data.append({
                        "Link Text": link_text,
                        "Target Page": target_page
                    })
            except Exception as e:
                print(f"Error extracting link on page {page_num} in {pdf_path}: {e}")
                continue

        page_num += 1
        if page_num >= doc.page_count:
            break

        try:
            next_page_text = doc.load_page(page_num).get_text("text")
        except Exception as e:
            print(f"Error loading next page {page_num} in {pdf_path}: {e}")
            break

        if not any(keyword in next_page_text for keyword in ["SECTION", "....", "INTRODUCTION"]):
            break

    if not toc_data:
        print(f"No TOC data extracted from {pdf_path}.")
        doc.close()
        return None

    df_links = pd.DataFrame(toc_data)

    def clean_text(text):
        text = re.sub(r'\.{2,}.*', '', text)
        text = text.strip()
        return text

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
    return toc_df

########################################
#         Extract Three Sections       #
########################################
def extract_three_sections(pdf_path, archive_dir):
    sections_to_extract = ["General Information", "Summary of the Offer Document", "Capital Structure"]

    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    toc_df = extract_table_of_contents(pdf_path)
    if toc_df is None or toc_df.empty:
        print(f"No TOC data available for {pdf_path}. Cannot extract sections.")
        return {section: None for section in sections_to_extract}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return {section: None for section in sections_to_extract}

    extracted_paths = {}

    for section_name in sections_to_extract:
        user_query_clean = section_name.lower().strip()
        subjects = toc_df['cleaned_subject'].tolist()
        subjects_lower = [s.lower() for s in subjects]

        if section_name.lower() == "general information":
            top_matches = process.extract(user_query_clean, subjects_lower, scorer=fuzz.partial_ratio, limit=2)
            if len(top_matches) == 0:
                print(f"No matches found for '{section_name}' in {pdf_path}.")
                extracted_paths[section_name] = None
            else:
                # second-best match logic for General Information
                if len(top_matches) > 1:
                    matched_subject_clean, score, match_index = top_matches[1]
                else:
                    matched_subject_clean, score, match_index = top_matches[0]

                matched_row = toc_df.iloc[match_index]
                start_page = int(matched_row['starting_page_number']) - 1
                end_page = int(matched_row['ending_page_number']) - 1

                if start_page < 0 or end_page >= doc.page_count or start_page > end_page:
                    print(f"Invalid page range for '{section_name}' in {pdf_path}.")
                    extracted_paths[section_name] = None
                else:
                    output_pdf_path = os.path.join(
                        archive_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{section_name.replace(' ', '_').lower()}.pdf"
                    )
                    try:
                        new_doc = fitz.open()
                        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                        new_doc.save(output_pdf_path)
                        new_doc.close()
                        print(f"Extracted '{section_name}' into '{output_pdf_path}'")
                        extracted_paths[section_name] = output_pdf_path
                    except Exception as e:
                        print(f"Error extracting '{section_name}': {e}")
                        extracted_paths[section_name] = None
        else:
            best_match = process.extractOne(user_query_clean, subjects_lower, scorer=fuzz.partial_ratio)
            if best_match:
                matched_subject_clean, score, match_index = best_match
                matched_row = toc_df.iloc[match_index]
                start_page = int(matched_row['starting_page_number']) - 1
                end_page = int(matched_row['ending_page_number']) - 1

                if start_page < 0 or end_page >= doc.page_count or start_page > end_page:
                    print(f"Invalid page range for '{section_name}' in {pdf_path}.")
                    extracted_paths[section_name] = None
                else:
                    output_pdf_path = os.path.join(
                        archive_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{section_name.replace(' ', '_').lower()}.pdf"
                    )
                    try:
                        new_doc = fitz.open()
                        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                        new_doc.save(output_pdf_path)
                        new_doc.close()
                        print(f"Extracted '{section_name}' into '{output_pdf_path}'")
                        extracted_paths[section_name] = output_pdf_path
                    except Exception as e:
                        print(f"Error extracting '{section_name}': {e}")
                        extracted_paths[section_name] = None
            else:
                print(f"No matches found for '{section_name}' in {pdf_path}.")
                extracted_paths[section_name] = None

    doc.close()
    return extracted_paths

########################################
#           PDF to Text Helper         #
########################################
def pdf_to_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return ""

    full_text = ""
    for page_num in range(doc.page_count):
        try:
            full_text += doc.load_page(page_num).get_text("text") + "\n"
        except Exception as e:
            print(f"Error extracting text from page {page_num} in {pdf_path}: {e}")
            continue
    doc.close()
    return full_text.strip()

########################################
#         General Info Prompt          #
########################################
general_info_prompt = '''
    Please carefully read the provided text and answer the following questions in the specified order:

    <question1>
    **Name of the Company**:  What is the name of the Company? Return only the name without any other additional information. 
    </question1>

    <question2>
    **IPO Grading**: Look for details about IPO Grading of the Issue. If No credit agency has been appointed for obtaining grading for the Offer (IPO), return 'No'. If details are found, return the. Return only the limited information
    If no credit agency has been appointed for obtaining grading for the Offer (IPO), return "No".
    </question2>

    <question3>
    **Underwriting Agreement**: Check for details about underwriting agreement of the Issue. There must be a clause which starts with something like "After determination of the Offer Price and allocation of Equity Shares, our Company enter into an Underwriting Agreement with the Underwriters for the Equity Shares proposed to be offered through the Offer. The Underwriting Agreement is dated [●]. Pursuant to the terms of the Underwriting Agreement, the obligations of each of the Underwriters will be several and will be subject to certain conditions specified therein
    If it's not found, return "Underwriting Details not found".
    </question3>

    <question4>
    **Abbreviation**: Abbreviate the name of the company using the initial letters of its name and return it.  For Example, Himalaya Computer Services Limited will be abbreviated as "HCSL". Return only the one word for the abbreviation and nothing else
    </question4>

    <question5>
    **Book Running Lead Managers**:  Book Running Lead Managers are Merchant Bankers registered with SEBI. There can be more than 1 Lead Managers. Find them and return their names separated by commas list. Please note that the names of the companies (Lead Managers) are needed, not their contact persons. Reutrn only the list of names and no extra information
    </question5>

    <question6>
    **Issue Type**: What is the type of issue? An issue can be of 3 types - "Fresh Issue", or "Offer for Sale" or "Fresh Issue and Offer for Sale". Return only the one word for the issue type and nothing else.
    </question6>

    Return the answers in JSON format with the following schema:
    {
    "underwriting_agreement": str,
    "ipo_grading": str,
    "issuer_name": str,
    "abbreviation": str,
    "lead_manager_list": str,
    "issue_type": str
    }

'''

########################################
#     Ask Questions for General Info   #
########################################
def ask_general_info_section(pdf_path, general_info_pdf_path):
    if not general_info_pdf_path or not os.path.exists(general_info_pdf_path):
        print("General Information PDF section not found.")
        return {}

    corpus_text = pdf_to_text(general_info_pdf_path)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return {}

    page_count = doc.page_count
    temp_dir = TEMP_DIR

    # Extract images of first two pages
    image1_path = os.path.join(temp_dir, "gen_info_page1.png")
    image2_path = os.path.join(temp_dir, "gen_info_page2.png")

    if page_count >= 1:
        try:
            pix = doc.load_page(0).get_pixmap(dpi=400)
            pix.save(image1_path)
        except Exception as e:
            print(f"Error extracting image from page 1: {e}")
    else:
        print("General Information section has less than 1 page.")

    if page_count >= 2:
        try:
            pix = doc.load_page(1).get_pixmap(dpi=400)
            pix.save(image2_path)
        except Exception as e:
            print(f"Error extracting image from page 2: {e}")

    doc.close()

    image1_encoded = encode_image(image1_path) if os.path.exists(image1_path) else ""
    image2_encoded = encode_image(image2_path) if os.path.exists(image2_path) else ""
    
    messages = [
        {
            "role": "system",
            "content": '''
                You are analyzing a section of a Draft Red Herring Prospectus (DRHP).
                The user provides the text from the "General Information" section and the questions.
                Respond with precise and brief answers in JSON format as requested.
            '''
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image1_encoded}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image2_encoded}"
                    }
                },
                {
                    "type": "text",
                    "text": corpus_text
                },
                {
                    "type": "text",
                    "text": general_info_prompt
                }
            ]
        }
    ]

    answer = ask_chatgpt(messages)
    try:
        parsed_answer = json.loads(answer)
        return parsed_answer
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON:", answer)
        return {"error": "Failed to parse JSON"}

########################################
#  Capital Structure Prompt            #
########################################
capital_structure_prompt = '''
    Please look at the images of the first two pages and the text from the first three pages of the DRHP's Capital Structure section, then answer the following questions in order:

    <question1> **Existing paid up Capital (Face Value)**: Look for details of Existing paid up capital of the company. The details are usually referenced as 'Equity Shares outstanding prior to the Offer' and after the conversion of any convertible shares. From the said information, extract face value of such shares along with the currency </question1>

    <question2> **Existing paid up Capital (No. of Shares)**:  Look for details of Existing paid up capital of the company. The details are usually referenced as 'Equity Shares outstanding prior to the Offer' and after the conversion of any convertible shares. From the said information, extract the number of shares.   </question2>
    
    <question3> **Existing paid up Capital (Nominal Amount)**: Look for details of Existing paid up capital of the company. The details are usually referenced as 'Equity Shares outstanding prior to the Offer' and after the conversion of any convertible shares. From the said information, extract the nominal amount (product of the existing face value and existing number of shares  as an integer </question3>

    <question4> **Existing paid up Capital (Existing Share Premium)**:  Look for details of Existing paid up capital of the company. The details are usually referenced as 'Equity Shares outstanding prior to the Offer' and after the conversion of any convertible shares. From the said information, extract the existing share premium along with the currency.  </question4>

    <question5> **Face Value**: The face value of Equity shares must be mentioned on the first page itself. It's ususally 1 rupee or 10 rupes. Check and express it in the format "Rs. 1/-" or "Rs. 10/-". Return only the face value along with currency  </question5>

    <question6> **Fresh Issue Size (No. of Shares)**: From the first page, read and tell me the number of shares of the fresh issue. The company may not be doing a fresh issue (only an offer for sale). In such case, it will be zero. Also, the shares could be undecided as on the date of filign of this doucment (this is represented by a symbol [●]. According to the details you have checked return the number of  shares, or zero if it's only OFS or [●] if that's indicated. </question6>

    <question7> **Fresh Issue Amount**:  Right. Fresh issue size is okay, now, From the first page, read and tell me the amount to be raised by the fresh issue. Usually this is mentioned as Size of Fresh issue: Upto [●] Equity shares of face value (some face value) aggregating upto ₹ (amont of fresh issue). The company may not be doing a fresh issue (only an offer for sale) in which case, fresh issue amount it zero. Or the company could be doing both. It is possible that the shares could be undecided as on the date of filing of this doucment - this is represented by a symbol [●]. According to the details you have checked return the amount raised by the fresh issue, or zero if it's only OFS or [●] if that's indicated. </question7>

    <question8> **OFS Size (No. of Shares)**:  From the first page, read and tell me the number of shares of the Offer For Sale. The company may be doing a fresh issue only. In such case, it will be zero. Or the company may do only an offer for sale or both.  Also, the shares could be undecided as on the date of filing of this doucment (this is represented by a symbol [●]. According to the details you have checked return the number of  shares, or zero if it's only fresh issue or [●] if that's indicated. </question8>

    <question9> **OFS Amount**: Right.After OFS shares, now, From the first page, read and tell me the amount to be raised by the OFS. Usually this is mentioned as Size of Offer For Sale: Upto (some number) Equity shares of face value (some face value) aggregating upto ₹ (amont of ofs). This amount of ofs is what needs to be extracted. The company may not be doing a an offer for sale (only fresh issue) in which case, fresh issue amount it zero. Or the company could be doing both. It is possible that the amount could be undecided as on the date of filing of this doucment - this is represented by a symbol [●]. According to the details you have checked return the amount raised by the OFS, or zero if it's only OFS or [●] if that's indicated. </question9>

    <question10> **Total Issue Size (No. of Shares)**: Right. After fresh issue and OFS shares, now, From the first page, read and tell me the total shares proposed to be issued (both fresh issue and OFS). it may be possible that either fresh issue or OFS may not be there, but at least one will be. Usually total issue size is mentioned as "Total Offer Size: Upto (some number) Equity shares of face value (some face value) aggregating upto ₹ (some amount)" in a table on the front page. This (some number) needs to be extracted as it represents the total shares to be issued. It is possible that the number of shares could be undecided as on the date of filing of this doucment - this is represented by a symbol [●]. According to the details you have checked return the shares or [●] if that's indicated. </question10>

    <question11> **Total Issue Amount**:  Right. After fresh issue and OFS shares, now, From the first page, read and tell me the total amount proposed to be raised (through both fresh issue and OFS). it may be possible that either fresh issue or OFS may not be there, but at least one will be. Usually total issue amount is mentioned as "Total Offer Size: Upto (some number) Equity shares of face value (some face value) aggregating upto ₹ (some amount)" in a table on the front page. This (some amount) needs to be extracted as it represents the total amount to be raised. It is possible that the total amount could be undecided as on the date of filing of this doucment - this is represented by a symbol [●]. According to the details you have checked return the amount with the currency and units or [●] if that's indicated. </question11>

    <question12> **Post Issue Share Premium**:  In the Offer Structure, you will find a line that says: "Offer of up to (some number of  [●]) Equity Shares of face value of ₹ (face value) each for cash at a price of ₹ (some number or [●]) per Equity Share (including a share premium of ₹ (some number or [●]) per Equity Share). This last amount where the share premium is either indicated or left incomplete with that symbol needs to be returned.  </question13>

    <question13> **POST ISSUE PAID UP CAPITAL (Face Value)**:  There's a table in the section titled Offer Structure. This table contains "ISSUED, SUBSCRIBED AND PAID-UP SHARE CAPITAL AFTER THE OFFER". Under this sub-section of the table, there will be details mentioned as "(some number or [●]) Equity Shares of face value of ₹ (some number or [●]) each." In the next cell, under the column Aggregate value at face value it will have "₹ (some number or [●])". Extract the value of  the second number (i.e. face value of shares outstanding after the offer) and return it. </question14>

    <question14> **POST ISSUE PAID UP CAPITAL (No. of Shares)**:  There's a table in the section titled Offer Structure. This table contains "ISSUED, SUBSCRIBED AND PAID-UP SHARE CAPITAL AFTER THE OFFER". Under this sub-section of the table, there will be details mentioned as "(some number or [●]) Equity Shares of face value of ₹ (some number or [●]) each." In the next cell, under the column Aggregate value at face value it will have "₹ (some number or [●])". Extract the value of  the first number (i.e. number of shares outstanding after the offer and return it. </question15>

    <question15> **POST ISSUE PAID UP CAPITAL (Nominal Amount)**: There's a table in the section titled Offer Structure. This table contains "ISSUED, SUBSCRIBED AND PAID-UP SHARE CAPITAL AFTER THE OFFER". Under this sub-section of the table, there will be details mentioned as "(some number or [●]) Equity Shares of face value of ₹ (some number or [●]) each." In the next cell, under the column Aggregate value at face value it will have "₹ (some number or [●])". Extract the value of  the last number (i.e. aggregate amount of share capital outstanding after the offer) and return it. </question16>

    Return answers in JSON:
    {
    "exist_face_value": str,
    "exist_num_shares": str,
    "exist_amount": str,
    "exist_share_premium": str,
    "issue_face_value": str,
    "fresh_issue_size": str,
    "fresh_issue_amount": str,
    "ofs_size": str,
    "ofs_amount": str,
    "total_issue_size": str,
    "total_issue_amount": str,
    
    "post_issue_share_premium": str,
    "post_face_value": str,
    "post_shares": str,
    "post_amount": str
    }
'''

########################################
#     Ask Questions for Capital Str    #
########################################
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def ask_capital_structure_section(pdf_path, capital_structure_pdf_path, temp_dir):
    if not capital_structure_pdf_path or not os.path.exists(capital_structure_pdf_path):
        print("Capital Structure PDF section not found.")
        return {}

    try:
        doc = fitz.open(capital_structure_pdf_path)
    except Exception as e:
        print(f"Error opening PDF {capital_structure_pdf_path}: {e}")
        return {}

    page_count = doc.page_count

    # Extract images of first two pages
    image1_path = os.path.join(temp_dir, "cap_str_page1.png")
    image2_path = os.path.join(temp_dir, "cap_str_page2.png")

    if page_count >= 1:
        try:
            pix = doc.load_page(0).get_pixmap(dpi=400)
            pix.save(image1_path)
        except Exception as e:
            print(f"Error extracting image from page 1: {e}")
    else:
        print("Capital Structure section has less than 1 page.")

    if page_count >= 2:
        try:
            pix = doc.load_page(1).get_pixmap(dpi=400)
            pix.save(image2_path)
        except Exception as e:
            print(f"Error extracting image from page 2: {e}")

    # Extract text from first three pages (if available)
    text_pages = []
    for i in range(min(page_count, 3)):
        try:
            text_pages.append(doc.load_page(i).get_text("text"))
        except Exception as e:
            print(f"Error extracting text from page {i+1}: {e}")
            continue
    doc.close()

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return {}

    page_count = doc.page_count

    # Extract images of first two pages
    image3_path = os.path.join(temp_dir, "page1.png")

    if page_count >= 1:
        try:
            pix = doc.load_page(0).get_pixmap(dpi=400)
            pix.save(image3_path)
        except Exception as e:
            print(f"Error extracting image from page 1: {e}")
    else:
        print("Office Note section has less than 1 page.")

    doc.close()

    combined_text = "\n\n".join(text_pages)

    image1_encoded = encode_image(image1_path) if os.path.exists(image1_path) else ""
    image2_encoded = encode_image(image2_path) if os.path.exists(image2_path) else ""
    image3_encoded = encode_image(image3_path) if os.path.exists(image3_path) else ""

    messages = [
        {
            "role": "system",
            "content": '''
                You are analyzing the Office Note section of a DRHP.
                The user provides images of the first two pages and text from the first three pages.
                Respond in JSON format as requested.
            '''
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image1_encoded}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image2_encoded}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image3_encoded}"
                    }
                },
                {
                    "type": "text",
                    "text": combined_text
                },
                {
                    "type": "text",
                    "text": capital_structure_prompt
                }
            ]
        }
    ]

    answer = ask_chatgpt(messages)
    try:
        parsed_answer = json.loads(answer)
        return parsed_answer
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON:", answer)
        return {"error": "Failed to parse JSON"}

########################################
#  Summary of the Offer Document Prompt#
########################################
summary_offer_prompt = '''
Please read the text from the "Summary of the Offer Document" section and answer the following questions in the specified order:

    <question1>
    **Objects of the Issue**: In the section of "Offer Document Summary", there will be a table about "Objects of the Offer" where the company says "Our Company proposes to utilise the Net Proceeds towards funding the following objects:". Extract the rows of this table as bullet points with amounts separated from the Particulars.
    </question1>

    <question2>
    **Industry**: From the "Offer Document Summary", return the sector/sub-sector of the Industry the company operates in. Use a few words only.
    </question2>

    <question3>
    **Rule 19(2) of SCRR check**: Check if exemption from Rule 19(2)(b) of SCRR is sought or if it says "If the Pre-IPO Placement is completed..." Return "Not Applicable" if it just says that the Pre-IPO placement will reduce the Fresh Issue (no exemption). If exemption details are given, return those details.
    </question3>

    <question4>
    **Exemption Details**: If any exemption or deviation or relaxation is sought, return the details. If no exemption is sought (confirmation shall be provided), return 'No exemption sought'.
    </question4>

    <question5>
    **Business of the Company**: In the Offer Document Summary, find details of the primary business of the Company and return that. Return what is provided in the document without summarizing it.
    </question5>

    <question6>
    **Pre Issue Promoter Shareholding**: There will be a section in the offer document summary that refers to Aggregate pre-Offer and post-Offer shareholding of our Promoters, members of our Promoter Group and Selling Shareholders as a percentage of our paid-up Equity Share capital. In this table, Percentage of pre-Offer paid-up Equity Share capital on a fully diluted basis (as a %) must be calculated. Please note that this is only for Promoters and Promoter Groups. Other Investor / selling shareholders need not be considered. Add up the total percentage value of the promoters' shareholding in the company prior to the issue and return that number along with the percentage sign
    </question6>

    <question7>
    **Post Issue Promoter Shareholding**: There will be a section in the offer document summary that refers to Aggregate pre-Offer and post-Offer shareholding of our Promoters, members of our Promoter Group and Selling Shareholders as a percentage of our paid-up Equity Share capital. In this table, Percentage of post-Offer paid-up Equity Share capital on a fully diluted basis (as a %) must be calculated. Please note that this is only for Promoters and Promoter Groups. Other Investor / selling shareholders need not be considered. Add up the total percentage value of the promoters' shareholding in the company prior to the issue and return that number along with the percentage sign. This may not be available sometimes and may be represented by [●]. In this case, return [●] only.
    </question7>

    Return the answers in JSON format:
    {
    "objects": str,
    "industry": str,
    "scrr_exemption": str,
    "exemption": str,
    "business": str,
    "pre_issue_promoter" : str,
    "post_issue_promoter" : str
    }
'''

########################################
#Ask Questions for Summary of Offer Doc#
########################################
def ask_summary_offer_document_section(pdf_path, summary_pdf_path):
    if not summary_pdf_path or not os.path.exists(summary_pdf_path):
        print("Summary of the Offer Document PDF section not found.")
        return {}

    corpus_text = pdf_to_text(summary_pdf_path)

    messages = [
        {
            "role": "system",
            "content": '''
                You are analyzing the "Summary of the Offer Document" section of a DRHP.
                The user provides the text from this section and the questions.
                Respond with precise and brief answers in JSON format as requested.
            '''
        },
        {
            "role": "user",
            "content": corpus_text
        },
        {
            "role": "user",
            "content": summary_offer_prompt
        }
    ]

    answer = ask_chatgpt(messages)
    try:
        parsed_answer = json.loads(answer)
        return parsed_answer
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON:", answer)
        return {"error": "Failed to parse JSON"}

########################################
#       Extract Subject as PDF         #
########################################
def extract_subject_pdf(pdf_path, df, user_query, output_pdf_path):
    if df.empty:
        print(f"No TOC data available for {pdf_path}. Skipping extraction.")
        return

    user_query_clean = user_query.lower().strip()
    subjects = df['cleaned_subject'].tolist()
    subjects_lower = [s.lower() for s in subjects]

    best_match = process.extractOne(user_query_clean, subjects_lower, scorer=fuzz.partial_ratio)
    
    if best_match:
        matched_subject_clean, score, match_index = best_match
        matched_row = df.iloc[match_index]
        matched_subject = matched_row['subject']
        matched_type = matched_row['Type']
        print(f"Selected Subject: '{matched_subject}' with a similarity score of {score}%")
        
        if matched_type == 'Section':
            try:
                start_page_str, end_page_str = matched_row['section_range'].split('-')
                start_page = int(start_page_str) - 1
                end_page = int(end_page_str) - 1
                print(f"Extracting entire section: {matched_subject} (pages {start_page + 1} to {end_page + 1})")
            except Exception as e:
                print(f"Error parsing section_range for '{matched_subject}': {e}")
                return
        else:
            try:
                start_page = int(matched_row['starting_page_number']) - 1
                end_page = int(matched_row['ending_page_number']) - 1
                print(f"Extracting subject: {matched_subject} (pages {start_page + 1} to {end_page + 1})")
            except Exception as e:
                print(f"Error parsing subject_range for '{matched_subject}': {e}")
                return

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            return

        if start_page < 0 or end_page >= doc.page_count or start_page > end_page:
            print(f"Invalid page range for '{matched_subject}': {start_page + 1} to {end_page + 1}")
            doc.close()
            return

        try:
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
            new_doc.save(output_pdf_path)
            new_doc.close()
            print(f"Extracted pages {start_page + 1} to {end_page + 1} into '{output_pdf_path}'")
        except Exception as e:
            print(f"Error extracting pages for '{matched_subject}': {e}")
        finally:
            doc.close()
    else:
        print(f"No matches found for the query '{user_query}' in {pdf_path}.")

########################################
#        Test General Info Code        #
########################################
# def test_general_info_extraction():
#     pdf_path = "backend\\vishal.pdf"
#     archive_dir = "backend\\archives"
    
#     sections = extract_three_sections(pdf_path, archive_dir)
#     general_info_pdf = sections.get("General Information")

#     results = ask_general_info_section(pdf_path, general_info_pdf)
#     print("General Information Q&A Results:")
#     print(json.dumps(results, indent=2, ensure_ascii=False))

########################################
#               MAIN                   #
########################################
# Uncomment the following lines to run tests directly
# if __name__ == "__main__":
#     test_general_info_extraction()
#     test_capital_structure_extraction()
#     test_summary_offer_extraction()
#     test_toc_2_page_qa()
#     test_means_of_finance()


########################################
#         TOC - 2 Page Q&A             #
########################################
toc_2_prompt = '''
    Please read the text from the first 2 pages of the DRHP (or the TOC section) and answer the following questions in the specified order:

    <question1>
    **Date of Incorporation**: This is a date on which the Issuer company was incorporated. Return the date in the format "month date (2 digit), Year (4 digit)". For example: October 04, 2024. Return only the date and nothing else.
    </question1>

    <question2>
    **In principle Approval**: On the first page, the Issuer must provide details of in-principle approval from BSE and / or NSE. Return 'BSE - In principle approval dated <Date>' and / or 'NSE - In principle approval dated <Date>' if dates are mentioned. If date not mentioned, return 'BSE - Yet to be received' or 'NSE - Yet to be received' accordingly. Separate multiple lines by a newline.
    </question2>

    <question3>
    **Net Offer Percentage**: On the second or third page, there will be a line: "THE OFFER AND THE NET OFFER SHALL CONSTITUTE (X%) AND (Y%) OF THE POST-OFFER PAID-UP EQUITY SHARE CAPITAL..." Extract Y% (the second percentage).
    </question3>

    <question4>
    **About the Company**: From the second or third page, return the paragraph (4-6 lines) about the company's history or incorporation. If it references another section, ignore that reference. Just return the paragraph text only.
    </question4>

    <question5>
    **Registered Office**: From the first page, return just the registered office address.
    </question5>

    Return in JSON format:
    {
    "date_of_incorp": str,
    "in_principle_approval": str,
    "net_public_offer": str,
    "about_company": str,
    "regd_office": str
    }
'''

def ask_toc_2_page_qa(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return {"error": "Failed to open PDF"}

    text_pages = []
    for i in range(min(doc.page_count, 2)):
        try:
            text_pages.append(doc.load_page(i).get_text("text"))
        except Exception as e:
            print(f"Error extracting text from page {i+1}: {e}")
            continue
    doc.close()

    corpus_text = "\n\n".join(text_pages)

    messages = [
        {
            "role": "system",
            "content": "You are analyzing the first two pages of a DRHP. The user asks certain questions based on this text."
        },
        {
            "role": "user",
            "content": corpus_text
        },
        {
            "role": "user",
            "content": toc_2_prompt
        }
    ]

    answer = ask_chatgpt(messages)
    try:
        parsed_answer = json.loads(answer)
        return parsed_answer
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON:", answer)
        return {"error": "Failed to parse JSON"}

########################################
#         Means of Finance Q&A         #
########################################
means_finance_prompt = '''
    "**Means of Finance**: As per the law, the issuer company must ensure that it has made arrangements of finance towards 75% of the stated means of finance for a specific project proposed to be funded from the issue proceeds, excluding the amount to be raised through the proposed public issue or through existing identifiable internal accruals.

    If any such information is provided, return it. If the issuer confirms that the Objects of the Issue are proposed to be funded from the Net Proceeds of the Offer only and no firm arrangements of finance are required under SEBI ICDR Regulations, return 'Not Applicable'.


    Return in JSON format:
    {
    "means_finance": str,
    }
'''

def ask_means_of_finance(pdf_path):
    # Extract full text
    full_text = pdf_to_text(pdf_path)

    # Search for all occurrences of "means of finance" case-insensitive
    occurrences = [m.start() for m in re.finditer(r'(?i)means of finance', full_text)]

    # For each occurrence, extract ±1000 chars of context
    context_snippets = []
    for occ in occurrences:
        start_index = max(0, occ - 1000)
        end_index = min(len(full_text), occ + 1000)
        snippet = full_text[start_index:end_index]
        context_snippets.append(snippet)

    # Combine all contexts
    combined_context = "\n\n".join(context_snippets) if context_snippets else ""

    # If no occurrences found, combined_context will be empty (which likely means "Not Applicable")
    # But we let LLM decide based on prompt
    if not combined_context.strip():
        combined_context = "No mention of means of finance found."

    messages = [
        {
            "role": "system",
            "content": "You are analyzing text related to 'Means of Finance' in a DRHP."
        },
        {
            "role": "user",
            "content": combined_context
        },
        {
            "role": "user",
            "content": means_finance_prompt
        }
    ]

    answer = ask_chatgpt(messages)
    try:
        parsed_answer = json.loads(answer)
        return parsed_answer
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON:", answer)
        return {"error": "Failed to parse JSON"}

########################################
#        Promoter Lock In              #
########################################

def ask_promoter_lock_in(pdf_path):
    # Extract full text

    doc = fitz.open(pdf_path)
    pages = [page for page in doc if "lock" in page.get_text("text").lower()]
    lock_in_text = "\n\n".join([page.get_text("text") for page in pages])
    doc.close()

    messages = [
        {
            "role": "system",
            "content": 
            '''
            The user will send you an excerpt from the DRHP filed by a company looking for raising capital through an IPO. You will analyze 
            the text and answer a question about the lock-in of shares of the promoters:

            **Promoter Lock in**: In terms of SEBI ICDR Regulations, the contribution of the promoters of a company is required to be lockedin for 
            some time. You will find this information in the section titled ""Details of Promoters' Contribution and lock-in. 
            It will look something like ""Pursuant to Regulations 14 and 16 of the SEBI ICDR Regulations, an aggregate of <percentage> of the fully 
            diluted post-Offer Equity Share capital of our Company held by the Promoters, shall be locked in for a period of <period> or any other 
            date as may be specified by SEBI as minimum promoter contribution from the date of Allotment, and the Promoter's shareholding in excess 
            of <percentage> of the fully diluted post-Offer Equity Share capital shall be locked in for a period of <period> or any other date as 
            may be specified by SEBI from the date of Allotment. Extract the entire clause and return it as a JSON object with the following schema:

            {
            "lockin": str,
            }
            '''
        },
        {
            "role": "user",
            "content": lock_in_text
        }
    ]

    answer = ask_chatgpt(messages)
    
    try:
        parsed_answer = json.loads(answer)
    
        return parsed_answer
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON:", answer)
        return {"error": "Failed to parse JSON"}

########################################
#               ENDPOINTS              #
########################################

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a file and extracts all requested information, returning them as one JSON."""
    # Clean or create temp folder
    temp_folder = TEMP_DIR
    if os.path.exists(temp_folder):
        for filename in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(temp_folder)

    # Sanitize the filename to prevent directory traversal
    sanitized_filename = os.path.basename(file.filename)
    pdf_path = os.path.join(temp_folder, sanitized_filename)  # Define pdf_path here
    
    # Save the uploaded file
    try:
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return JSONResponse(content={"error": "Failed to save uploaded file"}, status_code=500)

    # Extract the three sections
    archive_dir = os.path.join(temp_folder, "archives")
    sections = extract_three_sections(pdf_path, archive_dir)

    general_info_pdf = sections.get("General Information")
    summary_pdf = sections.get("Summary of the Offer Document")
    capital_str_pdf = sections.get("Capital Structure")

    reg_6_check, instrument, bidding_type = first_page_check(pdf_path)
    lockin = ask_promoter_lock_in(capital_str_pdf)
    
    # Run all Q&As
    # 1. General Info
    general_info_res = ask_general_info_section(pdf_path, general_info_pdf)

    # 2. Capital Structure
    # Create a temp_dir for images if needed
    image_temp_dir = os.path.join(temp_folder, "images")
    if not os.path.exists(image_temp_dir):
        os.makedirs(image_temp_dir)
    capital_res = ask_capital_structure_section(pdf_path, capital_str_pdf, image_temp_dir)

    # 3. Summary of the Offer Document
    summary_res = ask_summary_offer_document_section(pdf_path, summary_pdf)

    # 4. TOC - 2 page Q&A
    toc_2_res = ask_toc_2_page_qa(pdf_path)

    # 5. Means of Finance
    means_finance_res = ask_means_of_finance(pdf_path)

    combined_result = {
        **general_info_res,
        **capital_res,
        **summary_res,
        **toc_2_res,
        **means_finance_res,
        "reg_6_check": reg_6_check,
        "instrument": instrument,
        "bidding_type": bidding_type,
        **lockin,
    }

    print(combined_result)
    # # Merge all results into a single JSON
    # combined_result = {
    #     "general_info": general_info_res,
    #     "capital_structure": capital_res,
    #     "summary_offer_document": summary_res,
    #     "toc_2_page_qa": toc_2_res,
    #     "means_of_finance": means_finance_res
    # }

    return JSONResponse(content={
        "file_url": f"http://localhost:8000/temp/{os.path.basename(pdf_path)}",
        "extracted_info": combined_result
    })

@router.post("/upload_from_url")
async def upload_from_url(url: str = Query(...)):
    """Uploads a file from a URL and extracts all requested information."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error downloading PDF from URL: {e}")
        return JSONResponse(content={"error": "Failed to download PDF from URL"}, status_code=400)

    soup = BeautifulSoup(response.content, 'html.parser')
    iframe = soup.find('iframe')
    if iframe and 'src' in iframe.attrs:
        interimurl = iframe['src']
        pdf_url = interimurl.split('file=')[1] if 'file=' in interimurl else interimurl
    else:
        pdf_url = url  # Fallback if no iframe is found

    if not pdf_url.lower().endswith('.pdf'):
        print(f"URL does not point to a PDF file: {pdf_url}")
        return JSONResponse(content={"error": "URL does not point to a PDF file"}, status_code=400)

    temp_folder = TEMP_DIR
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    pdf_name = os.path.basename(pdf_url)
    pdf_path = os.path.join(temp_folder, pdf_name)

    try:
        with requests.get(pdf_url, stream=True) as r:
            r.raise_for_status()
            with open(pdf_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Error saving downloaded PDF: {e}")
        return JSONResponse(content={"error": "Failed to save downloaded PDF"}, status_code=500)

    # Extract the three sections
    archive_dir = os.path.join(temp_folder, "archives")
    sections = extract_three_sections(pdf_path, archive_dir)

    general_info_pdf = sections.get("General Information")
    summary_pdf = sections.get("Summary of the Offer Document")
    capital_str_pdf = sections.get("Capital Structure")

    # Run all Q&As
    general_info_res = ask_general_info_section(pdf_path, general_info_pdf)

    image_temp_dir = os.path.join(temp_folder, "images")
    if not os.path.exists(image_temp_dir):
        os.makedirs(image_temp_dir)
    capital_res = ask_capital_structure_section(pdf_path, capital_str_pdf, image_temp_dir)

    summary_res = ask_summary_offer_document_section(pdf_path, summary_pdf)
    toc_2_res = ask_toc_2_page_qa(pdf_path)
    means_finance_res = ask_means_of_finance(pdf_path)

    combined_result = {
        "general_info": general_info_res,
        "capital_structure": capital_res,
        "summary_offer_document": summary_res,
        "toc_2_page_qa": toc_2_res,
        "means_of_finance": means_finance_res
    }

    return JSONResponse(content=combined_result)

@router.post("/generate_office_note")
async def generate_office_note(info: dict):
    """Generates an office note based on the extracted information."""
    source_file = r'D:\drhp_procesing\backend\files\OfficeNote.docx'
    destination_file = os.path.join(TEMP_DIR, 'GeneratedOfficeNote.docx')

    # Copy the source file to the destination file
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    shutil.copyfile(source_file, destination_file)

    # Replace placeholders in the document
    doc = DocxTemplate(destination_file)

    # Ensure the context is aligned with the frontend variables
    context = {
        'issuer_name': info.get('issuer_name', 'N/A'),
        'abbreviation': info.get('abbreviation', 'N/A'),
        'lead_manager_list': info.get('lead_manager_list', 'N/A'),
        'date_of_incorp': info.get('date_of_incorp', 'N/A'),
        'issue_type': info.get('issue_type', 'N/A'),
        'instrument': info.get('instrument', 'N/A'),
        'bidding_type': info.get('bidding_type', 'N/A'),
        'underwriting_agreement': info.get('underwriting_agreement', 'N/A'),
        'exist_face_value': info.get('exist_face_value', 'N/A'),
        'exist_num_shares': info.get('exist_num_shares', 'N/A'),
        'exist_amount': info.get('exist_amount', 'N/A'),
        'exist_share_premium': info.get('exist_share_premium', 'N/A'),
        'issue_face_value': info.get('issue_face_value', 'N/A'),
        'fresh_issue_size': info.get('fresh_issue_size', 'N/A'),
        'fresh_issue_amount': info.get('fresh_issue_amount', 'N/A'),
        'ofs_size': info.get('ofs_size', 'N/A'),
        'ofs_amount': info.get('ofs_amount', 'N/A'),
        'total_issue_amount': info.get('total_issue_amount', 'N/A'),
        'post_issue_share_premium': info.get('post_issue_share_premium', 'N/A'),
        'total_issue_size': info.get('total_issue_size', 'N/A'),
        'objects': info.get('objects', 'N/A'),
        'means_finance': info.get('means_finance', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'scrr_exemption': info.get('scrr_exemption', 'N/A'),
        'pre_issue_promoter': info.get('pre_issue_promoter', 'N/A'),
        'post_issue_promoter': info.get('post_issue_promoter', 'N/A'),
        'net_public_offer': info.get('net_public_offer', 'N/A'),
        'lockin': info.get('lockin', 'N/A'),
        'ipo_grading': info.get('ipo_grading', 'N/A'),
        
        'post_face_value': info.get('post_face_value', 'N/A'),
        'post_shares': info.get('post_shares', 'N/A'),
        'post_amount': info.get('post_amount', 'N/A'),
        'exemption': info.get('exemption', 'N/A'),
        'business': info.get('business', 'N/A'),
        'in_principle_approval': info.get('in_principle_approval', 'N/A'),
        'about_company': info.get('about_company', 'N/A'),
        'regd_office': info.get('regd_office', 'N/A'),
    }

    doc.render(context)
    doc.save(destination_file)

    return FileResponse(destination_file, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

@router.get("/temp/{filename}")
async def get_temp_file(filename: str):
    """Retrieves a temporary file by filename."""
    file_path = os.path.join("temp", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"detail": "Not Found"}, status_code=404)