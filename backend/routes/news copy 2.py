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
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

router = APIRouter()

# Helper Functions

class Message(BaseModel):
    role: str
    content: str

class CustomQueryInput(BaseModel):
    user_input: str

def fetch_ipo_news(company_name: str):
    queries = {
        "News about the Company's IPO": f"Find and summarize the latest news articles about the upcoming IPO of {company_name} in India. Focus on any potential controversies or legal scrutiny involving the company, its promoters, or its directors.",
        "Interviews by Directors": f"Search for and provide a summary of interviews given by the directors or promoters of {company_name} regarding its IPO.",
        "Litigation Cases": f"Identify and summarize any outstanding litigation cases involving {company_name}.",
        "Criminal and Civil Actions": f"Search for criminal or civil actions taken against {company_name} or its directors, promoters or Key Executives.",
        "Regulatory Complaints": f"Search for any complaints filed against {company_name}, its promoters, directors or key executives."
    }

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    results = []
    for title, query in queries.items():
        try:
            data = {
                "model": "pplx-7b-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that searches for and summarizes news articles."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }

            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                result = response_data["choices"][0]["message"]["content"]
                citations = []

                # Extract URLs using regex
                url_pattern = re.compile(
                    r'(?:https?:\/\/|www\.)[^\s<>"]+?(?:\([\w\d]+\)|([^<>"\')\s]|\/)*)'
                )
                citations = url_pattern.findall(result)
                citations = [url for url in citations if url]  # Remove empty matches

                results.append({
                    "title": title,
                    "query": query,
                    "result": result,
                    "citations": citations
                })
            else:
                results.append({
                    "title": title,
                    "query": query,
                    "result": "No response received from the API",
                    "citations": []
                })

        except Exception as e:
            results.append({
                "title": title,
                "query": query,
                "result": f"An error occurred: {e}",
                "citations": []
            })

    small_results = ""
    for result in results:
        small_results += json.dumps({
            "title": result["title"],
            "result": result["result"]
        }) + "\n"

    gpt_response = ask_gpt(small_results)
    
    return results, gpt_response

def custom_query(user_input: str):
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
                "content": "Be accurate and detialed."
            },
            {
                "role": "user",
                "content": user_input
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
        return {"result": result, "citations": data.get("citations")}
        
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

def ask_gpt(abc: str):
    messages = [{
        "role": "system",
        "content": '''
                The user will provide you with a list of news articles and their summaries 
                generated using LLMs. Analyze the articles and provide a list of Risks 
                about the company which is planning to raise funds through an IPO.
                Risks include litigations, lawsuits, legal actions, and other complaints.
                Don’t be too speculative. Lack of specific complaints does not indicate a risk by itself.
                Mention only if something specific is found. No need to mention anything if no specific
                complaints are found.
                
        '''
    },
    {
        "role": "user",
        "content": abc
    }]
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while analyzing the text: {e}"

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

def google_search_and_summarize(q: str, max_results=5):
    excluded_exts = ["pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx"]
    ex_q = " ".join([f"-filetype:{ext}" for ext in excluded_exts])
    mod_q = f"{q} {ex_q}"
    print(f"[INFO] Searching google: {mod_q}")

    try:
        # Just get URLs without processing
        found_urls = list(search(mod_q, num=max_results, pause=1))
        return {
            "urls": found_urls,
            "query": q
        }
    except Exception as ex:
        print(f"[ERROR] Google search failed: {ex}")
        return {"error": str(ex)}

def gpt_search(company_name: str):
    queries = [
        {
            "query": f"(litigation OR legal case OR criminal) AND ({company_name})",
            "label": "Legal & Regulatory Issues"
        },
        {
            "query": f"(SEBI OR RBI OR MCA OR SFIO OR ED) AND action AND ({company_name})",
            "label": "Regulatory Actions"
        },
        {
            "query": f"(penalty OR debarment OR fraud OR warning) AND ({company_name})",
            "label": "Penalties & Warnings"
        }
    ]

    results = []
    for q in queries:
        print(f"=== GPT-based search for question: {q['query']}")
        try:
            search_result = google_search_and_summarize(q['query'], max_results=5)
            if isinstance(search_result, dict):
                if "error" in search_result:
                    results.append({
                        "query": q['query'],
                        "label": q['label'],
                        "urls": [],
                        "error": search_result["error"]
                    })
                else:
                    results.append({
                        "query": q['query'],
                        "label": q['label'],
                        "urls": search_result["urls"],
                        "error": None
                    })
        except Exception as e:
            print(f"Error processing query '{q}': {e}")
            results.append({
                "query": q['query'],
                "label": q['label'],
                "urls": [],
                "error": str(e)
            })

    return results

@router.get('/fetch_ipo_news')
def get_ipo_news(company_name: str = Query(...)):
    try:
        # Get results from Perplexity API
        perplexity_results, perplexity_red_flags = fetch_ipo_news(company_name)
        
        # Get Google search results
        gpt_search_results = gpt_search(company_name)
        
        return {
            "perplexity_results": perplexity_results,
            "perplexity_red_flags": perplexity_red_flags,
            "google_results": gpt_search_results,
            "gpt_red_flags": "Analysis in progress..."
        }
    except Exception as e:
        print(f"[ERROR] Error in get_ipo_news: {str(e)}")
        return {
            "error": f"An error occurred: {str(e)}",
            "perplexity_results": [],
            "perplexity_red_flags": "Error fetching red flags",
            "google_results": [],
            "gpt_red_flags": "Error in GPT analysis"
        }

@router.post("/custom_query")
def post_custom_query(input_data: CustomQueryInput):
    result = custom_query(input_data.user_input)
    return result