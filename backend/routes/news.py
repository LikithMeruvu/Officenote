# import os
# import requests
# import openai
# from openai import OpenAI
# from fastapi import Query, Body, APIRouter, HTTPException
# from typing import List, Dict, Any
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import json

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PERPLEXITY_API_KEY = "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"

# openai.api_key = OPENAI_API_KEY

# router = APIRouter()

# class Message(BaseModel):
#     role: str
#     content: str

# class CustomQueryInput(BaseModel):
#     user_input: str

# def fetch_ipo_news(company_name: str):
#     queries = {
#         "News about the Company's IPO": f"Find and summarize the latest news articles about the upcoming IPO of {company_name} in India. Focus on any potential controversies or legal scrutiny involving the company, its promoters, or its directors.",
#         "Interviews by Directors": f"Search for and provide a summary of interviews given by the directors or promoters of {company_name} regarding its IPO. Focus on their statements about the company's financial health, business model, and compliance with regulatory requirements.",
#         "Legal Troubles with Directors": f"Search for any legal troubles or lawsuits involving the directors of {company_name}, which is planning to go public with its IPO in India. Include any past or pending cases that could impact the company's reputation.",
#         "Litigation Cases": f"Identify and summarize any outstanding litigation cases involving {company_name}, especially those related to financial mismanagement, regulatory issues, or shareholder disputes.",
#         "Promoter Legal Actions": f"Search for legal actions or investigations against the promoters of {company_name} company, which is coming for its IPO. Include criminal charges, fraud allegations, or regulatory breaches in your findings.",
#         "Criminal and Civil Actions": f"Search for criminal or civil actions taken against both the promoters and directors of {company_name}, focusing on cases that could affect the credibility of their upcoming IPO in India.",
#         "Mismanagement Reports": f"Provide a summary of any news articles reporting on potential mismanagement at {company_name}, especially those highlighting operational failures or leadership issues at the promoter or director level.",
#         "Financial Inaccuracies": f"Search for any news items or reports suggesting inaccuracies or inconsistencies in the financial statements of {company_name}, with particular attention to any discrepancies that could affect its IPO prospects.",
#         "Corporate Governance Issues": f"Identify and summarize news related to corporate governance issues at {company_name}, including concerns raised about the management, ethical practices, or transparency of the company as it prepares for its IPO.",
#         "Analyst Insights": f"Find recent market analyst reports or insights about {company_name}'s upcoming IPO. Focus on their evaluation of the company's financials, growth potential, and any red flags that analysts have identified.",
#         "Regulatory Complaints": f"Search for any consumer or regulatory complaints filed against {company_name}, its promoters, or its directors. Include complaints related to financial disclosures, operational practices, or misrepresentation."
#     }

#     url = "https://api.perplexity.ai/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     results = []

#     for title, query in queries.items():
#         payload = {
#             "model": "llama-3.1-sonar-small-128k-online",
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": "This is for news report generation. Keep yourself brief and accurate. Stay on the subject. Do not add unnecessary stuff."
#                 },
#                 {
#                     "role": "user",
#                     "content": query
#                 }
#             ],
#             "max_tokens": 4000,
#             "temperature": 0,
#             "top_p": 0.9,
#             "return_citations": True,
#             "search_domain_filter": ["perplexity.ai"],
#             "return_images": False,
#             "return_related_questions": False,
#             "search_recency_filter": "month",
#             "top_k": 0,
#             "stream": False,
#             "presence_penalty": 0,
#             "frequency_penalty": 1
#         }

#         try:
            
#             response = requests.post(url, json=payload, headers=headers)
#             response.raise_for_status()
#             data = response.json()
#             result = data.get("choices", [{}])[0].get("message", {}).get("content", "No news available.")
#             citations = data.get("citations", [])
#             results.append({
#                 "title": title,
#                 "query": query,
#                 "result": result,
#                 "citations": citations
#             })
          
            
#         except requests.exceptions.RequestException as e:
#             results.append({
#                 "title": title,
#                 "query": query,
#                 "result": f"An error occurred: {e}",
#                 "citations": []
#             })
          

#     small_results = ""
#     for result in results:
#         small_results += json.dumps({
#             "query": query,
#             "result": result,
#         }) + "\n"

#     gpt_response = ask_gpt(small_results)

#     print("Results being returned:", results)  # Log the results before returning
#     return results, gpt_response

# def custom_query(user_input: str):
#     url = "https://api.perplexity.ai/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "llama-3.1-sonar-small-128k-online",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "Be accurate and detialed."
#             },
#             {
#                 "role": "user",
#                 "content": user_input
#             }
#         ],
#         "max_tokens": 4000,
#         "temperature": 0,
#         "top_p": 0.9,
#         "return_citations": True,
#         "search_domain_filter": ["perplexity.ai"],
#         "return_images": False,
#         "return_related_questions": False,
#         "search_recency_filter": "month",
#         "top_k": 0,
#         "stream": False,
#         "presence_penalty": 0,
#         "frequency_penalty": 1
#     }

#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         response.raise_for_status()
#         data = response.json()
#         result = data.get("choices", [{}])[0].get("message", {}).get("content", "No news available.")
#         return {"result": result, "citations": data.get("citations")}
        
#     except requests.exceptions.RequestException as e:
#         return f"An error occurred: {e}"

# def ask_gpt(abc: str):
#     messages = [{
#         "role": "system",
#         "content": '''
#                 The user will provide you with a list of news articles and their summaries 
#                 generated using LLMs. Analyze the articles and provide a list of Risks 
#                 about the company which is planning to raise funds through an IPO.
#                 Risks include litigations, lawsuits, legal actions, and other complaints.
#                 Don't be too speculative. Lack of specific complaints does not indicate a risk by itself.
#                 Mention only if something specific is found. No need to mention if no specific
#                 complaints are found.
                
#         '''
#     },
#     {
#         "role": "user",
#         "content": abc
#     }]
    
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             temperature=0,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"An error occurred while analyzing the text: {e}"

# def analyze_red_flags(messages: List[Dict[str, str]]):
#     # NOTE: If you're getting an error related to ChatCompletion, make sure you have
#     # a compatible version of the openai library. For example:
#     # pip install openai==0.27.0
#     # Or migrate your code with openai migrate.
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",
#             messages=messages
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"An error occurred while analyzing the text: {e}"

# @router.get('/fetch_ipo_news')
# def get_ipo_news(company_name: str = Query(...)):
#     print("Processing news request for:", company_name)
#     results, gpt_response = fetch_ipo_news(company_name)
#     print(gpt_response)
#     return {"results" : results, "gpt_response": gpt_response}

# @router.post("/custom_query")
# def post_custom_query(input_data: CustomQueryInput):
#     result = custom_query(input_data.user_input)
#     return result








import os
import json
import re
import time
import requests
from newspaper import Article, Config
from googlesearch import search
from openai import OpenAI
from fastapi import FastAPI, APIRouter, Query
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = "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"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Create the FastAPI app
app = FastAPI()

router = APIRouter()

###############################################################################
# 1) ask_gpt: Summarize red flags from combined text
###############################################################################
def ask_gpt(abc: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. Analyze the provided news articles and list any possible risks or red flags about a company planning an IPO. "
                "Mention specific items if found; otherwise, state that none were found. Be diplomatic and professional in your responses."
            )
        },
        {
            "role": "user",
            "content": abc
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while analyzing the text: {e}"

###############################################################################
# 2) Perplexity-based 10 IPO queries
###############################################################################
def fetch_ipo_news(company_name: str):
    queries = {
        "News about the Company's IPO": f"Find and summarize the latest news articles about the upcoming IPO of {company_name} in India. Focus on any potential controversies or legal scrutiny involving the company, its promoters, or its directors.",
        "Interviews by Directors": f"Search for and provide a summary of interviews given by the directors or promoters of {company_name} regarding its IPO. Focus on their statements about the company's financial health, business model, and compliance with regulatory requirements.",
        "Legal Troubles with Directors": f"Search for any legal troubles or lawsuits involving the directors of {company_name}, which is planning to go public with its IPO in India. Include any past or pending cases that could impact the company's reputation.",
        "Litigation Cases": f"Identify and summarize any outstanding litigation cases involving {company_name}, especially those related to financial mismanagement, regulatory issues, or shareholder disputes.",
        "Promoter Legal Actions": f"Search for legal actions or investigations against the promoters of {company_name} company, which is coming for its IPO. Include criminal charges, fraud allegations, or regulatory breaches in your findings.",
        "Criminal and Civil Actions": f"Search for criminal or civil actions taken against both the promoters and directors of {company_name}, focusing on cases that could affect the credibility of their upcoming IPO in India.",
        "Mismanagement Reports": f"Provide a summary of any news articles reporting on potential mismanagement at {company_name}, especially those highlighting operational failures or leadership issues at the promoter or director level.",
        "Financial Inaccuracies": f"Search for any news items or reports suggesting inaccuracies or inconsistencies in the financial statements of {company_name}, with particular attention to any discrepancies that could affect its IPO prospects.",
        "Corporate Governance Issues": f"Identify and summarize news related to corporate governance issues at {company_name}, including concerns raised about the management, ethical practices, or transparency of the company as it prepares for its IPO.",
        "Analyst Insights": f"Find recent market analyst reports or insights about {company_name}'s upcoming IPO. Focus on their evaluation of the company's financials, growth potential, and any red flags that analysts have identified.",
        "Regulatory Complaints": f"Search for any consumer or regulatory complaints filed against {company_name}, its promoters, or its directors. Include complaints related to financial disclosures, operational practices, or misrepresentation."
    }

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    results = []

    for title, query in queries.items():
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "This is for news report generation. Keep yourself brief and accurate. Stay on the subject. Do not add unnecessary stuff."
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
            resp = requests.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            result_text = data.get("choices", [{}])[0].get("message", {}).get("content", "No news available.")
            cits = data.get("citations", [])
            results.append({
                "title": title,
                "query": query,
                "result": result_text,
                "citations": cits
            })
        except requests.exceptions.RequestException as e:
            results.append({
                "title": title,
                "query": query,
                "result": f"An error occurred: {e}",
                "citations": []
            })

    combined_text = ""
    for r in results:
        combined_text += f"Title: {r['title']}\nQuery: {r['query']}\nResult: {r['result']}\n\n"

    gpt_response = ask_gpt(combined_text)
    return results, gpt_response

###############################################################################
# 3) GPT-based local google search: 6 queries
###############################################################################
def fetch_html_and_summarize(url, head_timeout=5, parse_timeout=5):
    try:
        hr = requests.head(url, timeout=head_timeout, allow_redirects=True)
        ctype = hr.headers.get("Content-Type", "").lower()
        if "html" not in ctype:
            print(f"Skipping non-HTML: {url}, ctype={ctype}")
            return None
    except:
        return None

    conf = Config()
    conf.request_timeout = parse_timeout
    conf.browser_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    )
    start_t = time.time()
    try:
        article = Article(url, config=conf)
        article.download()
        if (time.time() - start_t) > parse_timeout:
            print(f"Skipping {url}, download took too long.")
            return None
        article.parse()
        if (time.time() - start_t) > parse_timeout:
            print(f"Skipping {url}, parse took too long.")
            return None

        return {
            "url": url,
            "text": article.text
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def google_search_and_summarize(q: str, max_results=15):
    excluded_exts = ["pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx"]
    ex_q = " ".join([f"-filetype:{ext}" for ext in excluded_exts])
    mod_q = f"{q} {ex_q}"
    print(f"[INFO] Searching google: {mod_q}")

    ret_list = []
    try:
        found_urls = search(mod_q, num_results=max_results, sleep_interval=3)
    except Exception as ex:
        return json.dumps({"error": str(ex)})

    seen = set()
    for url in found_urls:
        if any(url.lower().endswith(ext) for ext in excluded_exts):
            continue
        if url in seen:
            continue
        seen.add(url)

        item = fetch_html_and_summarize(url)
        if item:
            ret_list.append(item)

    return ret_list

def gpt_search(company_name: str):
    queries = [
        f"default attachment litigation case order related to {company_name}",
        f"criminal litigation IPC CBI SFIO Enforcement Directorate action against {company_name}",
        f"MCA ROC SEBI RBI PMLA actions orders against {company_name}",
        f"income tax penalty, search and seizure, debarment of {company_name}",
        f"USFDA actions, SEC, Ministry action, legal troubles, about {company_name}",
        f"Fraud, illegal action, warning, litigation, suspension of {company_name}"
    ]

    out_list = []
    for q in queries:
        print(f"=== GPT-based search for question: {q}")
        articles = google_search_and_summarize(q, max_results=3)
        if isinstance(articles, list):
            lines = [f"Article {idx+1} URL: {art['url']}\nText: {art['text']}" for idx, art in enumerate(articles)]
            combined_text = "\n".join(lines)
        else:
            combined_text = f"Error from search: {articles.get('error', 'Unknown error')}"

        sys_msg = {
            "role": "system",
            "content": (
                "You are a helpful AI. Summarize relevant info about possible red flags or litigations. "
                "Include references to any relevant URLs. Be diplomatic and professional in your responses."
            )
        }
        user_msg = {
            "role": "user",
            "content": f"Here is context:\n\n{combined_text}\n\nQuestion: {q}"
        }

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[sys_msg, user_msg],
                temperature=0
            )
            final_text = resp.choices[0].message.content
        except Exception as e:
            final_text = f"Error calling GPT: {e}"

        url_pattern = re.compile(
            r"(?:https?:\/\/|ftp:\/\/|www\.)[\w\-_]+(?:\.[\w\-_]+)+(?:[\w\.,@?^=%&:/~+#\-\(\)]*)?"
        )
        cits = url_pattern.findall(final_text)

        out_list.append({
            "question": q,
            "final_response": final_text,
            "citations": cits
        })

    return out_list

###############################################################################
# 4) Combining perplexity (10) + local GPT search (6) => final
###############################################################################
def combined_research(company_name: str):
    # Fetch Perplexity results
    perplex_res, perplex_red_flags = fetch_ipo_news(company_name)

    # Fetch GPT-based local search results
    gpt_res = gpt_search(company_name)

    # Combine all text from Perplexity and GPT search results
    combined_text = "=== Perplexity Results ===\n"
    for r in perplex_res:
        combined_text += f"Title: {r['title']}\nQuery: {r['query']}\nResult: {r['result']}\n\n pplx_red_flag: {perplex_red_flags}"

    combined_text += "\n=== GPT Search Results ===\n"
    for res in gpt_res:
        combined_text += f"Question: {res['question']}\n"
        for idx, art in enumerate(res.get("articles", [])):
            combined_text += f"Article {idx+1} URL: {art['url']}\nText: {art['text']}\n\n"

    # Identify red flags from combined text
    final_red_flags = ask_gpt(combined_text)

    return {
        "perplexity_results": perplex_res,
        "gpt_results": gpt_res,
        "combined_red_flags": final_red_flags
    }

###############################################################################
# 5) FastAPI endpoint
###############################################################################
@router.get("/fetch_ipo_news")
def get_ipo_news(company_name: str = Query(...)):
    """
    1) Perplexity 10 queries => perplexity_results + short perplexity_red_flags
    2) Local GPT-based google => 6 queries => gpt_results
    3) Combine => final red flags => returned
    """
    out = combined_research(company_name)
    return out

app.include_router(router)