from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import pandas as pd
import tempfile
import os
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# Hardcoded metadata for the five Excel files
EXCEL_FILES_METADATA = [
    {
        "Filename": "excel_1.xlsx",
        "title": "List of cases dismissed in the prosecution filed by SEBI",
        "pdf_link": "https://www.sebi.gov.in/sebi_data/attachdocs/apr-2023/1682506243254.pdf",
        "file_path": "C:/Users/manoj/Downloads/cfd/drhp/app/backend/excels/excel_1.xlsx"
    },
    {
        "Filename": "excel_2.xlsx",
        "title": "List of cases in which accused declared as Proclaimed Offenders in the prosecution filed by SEBI",
        "pdf_link": "https://www.sebi.gov.in/sebi_data/attachdocs/jun-2021/1624881566107.pdf",
        "file_path": "C:/Users/manoj/Downloads/cfd/drhp/app/backend/excels/excel_2.xlsx"
    },
    {
        "Filename": "excel_3.xlsx",
        "title": "List of Cases resulted in compounding in the prosecution filed by SEBI",
        "pdf_link": "https://www.sebi.gov.in/sebi_data/attachdocs/apr-2023/1682506150614.pdf",
        "file_path": "C:/Users/manoj/Downloads/cfd/drhp/app/backend/excels/excel_3.xlsx"
    },
    {
        "Filename": "excel_4.xlsx",
        "title": "List of cases resulted in convictions in the prosecution filed by SEBI",
        "pdf_link": "https://www.sebi.gov.in/sebi_data/attachdocs/oct-2018/1539334171190.pdf",
        "file_path": "C:/Users/manoj/Downloads/cfd/drhp/app/backend/excels/excel_4.xlsx"
    },
    {
        "Filename": "excel_5.xlsx",
        "title": "LIST OF DEFAULTERS AS ON MAY 31, 2018 FOR NON-PAYMENT OF PENALTY IMPOSED BY SEBI THROUGH ORDERS PASSED UPTO DECEMBER 31, 2017",
        "pdf_link": "https://www.sebi.gov.in/sebi_data/attachdocs/apr-2023/1682506275026.pdf",
        "file_path": "C:/Users/manoj/Downloads/cfd/drhp/app/backend/excels/excel_5.xlsx"
    },
    {
        "Filename": "excel_6.xlsx",
        "title": "List of Vanishing Companies",
        "pdf_link": "",
        "file_path": "C:/Users/manoj/Downloads/cfd/drhp/app/backend/excels/excel_6.xlsx"
    }
]

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)

def clean_nan_values(obj):
    if isinstance(obj, float) and pd.isna(obj):
        return None
    elif isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d')
    return obj

def search_names_in_dataframe(df: pd.DataFrame, user_names: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Searches every cell in each DataFrame for any of the user_names as a substring (case-insensitive).
    If a match is found, returns a dictionary that includes:
    - The headers of the DataFrame
    - The entire matched row
    - The excel_file_name
    - The pdf_link
    - The matched_user_name
    - The matched_pdf_value (the cell that contained the matched name)
    """
    results = []
    for idx, row in df.iterrows():
        # Convert entire row to strings (lowercase)
        row_str_values = [str(val).lower() if pd.notna(val) else "" for val in row.values]
        matched_user_name = None
        matched_pdf_value = None

        # Check for direct substring matches (case-insensitive)
        for uname in user_names:
            uname_lower = uname.lower()
            for cell_val in row_str_values:
                if uname_lower in cell_val:
                    # Found a match
                    matched_user_name = uname
                    matched_pdf_value = cell_val
                    break
            if matched_user_name:
                break

        if matched_user_name and matched_pdf_value:
            # Store entire row data as a dict
            row_data = row.to_dict()
            # Create the result entry
            entry = {
                'headers': df.columns.tolist(),
                'row_data': row_data,
                'excel_file_name': metadata['title'],
                'pdf_link': metadata['pdf_link'],
                'matched_user_name': matched_user_name,
                'matched_pdf_value': matched_pdf_value
            }
            results.append(entry)
    return results

@router.post("/db-check")
async def process_excel(file: UploadFile = File(...)):
    logging.info(f'Received file: {file.filename}')
    if not file.filename.endswith(('.xlsx', '.xls')):
        logging.error('Invalid file type. Please upload an Excel file.')
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an Excel file (.xlsx or .xls).")

    try:
        # Save the uploaded file temporarily
        contents = await file.read()
        logging.info('File read successfully.')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Read the uploaded Excel file into pandas DataFrame
            names_df = pd.read_excel(tmp_path)
            logging.info('Excel file read into DataFrame.')
        except Exception as e:
            logging.error(f'Failed to read the uploaded Excel file: {e}')
            raise HTTPException(status_code=400, detail=f"Failed to read the uploaded Excel file: {e}")
        finally:
            os.unlink(tmp_path)  # Delete the temporary file

        if names_df.empty:
            logging.error('The uploaded Excel file is empty.')
            raise HTTPException(status_code=400, detail="The uploaded Excel file is empty.")

        # Extract names from the first column
        first_column = names_df.columns[0]
        user_names = names_df[first_column].dropna().unique().tolist()

        if not user_names:
            logging.error('No names found in the first column of the uploaded Excel file.')
            raise HTTPException(status_code=400, detail="No names found in the first column of the uploaded Excel file.")

        # Initialize a list to collect all matches
        all_matches = []

        # Iterate over the hardcoded Excel files
        for metadata in EXCEL_FILES_METADATA:
            excel_path = metadata['file_path']

            # Check if the file exists
            if not os.path.isfile(excel_path):
                logging.error(f"Data file '{metadata['Filename']}' does not exist at path: {excel_path}")
                raise HTTPException(status_code=400, detail=f"Data file '{metadata['Filename']}' does not exist at path: {excel_path}")

            try:
                # Read the Excel file into a pandas DataFrame
                data_df = pd.read_excel(excel_path)
                logging.info(f"Read data file: {metadata['Filename']}")
            except Exception as e:
                logging.error(f"Failed to read data file {metadata['Filename']}: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to read data file '{metadata['Filename']}': {e}")

            if data_df.empty:
                continue  # Skip empty files

            # Search for names in the DataFrame
            matches = search_names_in_dataframe(data_df, user_names, metadata)
            all_matches.extend(matches)

        # Clean nan values before JSON serialization
        cleaned_matches = clean_nan_values(all_matches)
        
        # Use custom JSON encoder for datetime objects
        json_data = json.dumps({"matches": cleaned_matches}, cls=DateTimeEncoder)
        
        return JSONResponse(content=json.loads(json_data))

    except HTTPException as he:
        raise he  # Re-raise HTTP exceptions to be handled by FastAPI
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))