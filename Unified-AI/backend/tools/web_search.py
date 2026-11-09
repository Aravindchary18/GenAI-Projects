import os

from langchain_tavily import TavilySearch

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("TAVILY_AKEY"):

    raise ValueError(
        "TAVILY_API_KEY not found in .env"
    )
    
web_search_tool = TavilySearch(
        tavily_api_key = os.getenv("TAVILY_AKEY"),
        max_results=3,
        topic="general",
        search_depth="fast",
        timeout = 180
    )
