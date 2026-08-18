import os

from langchain_tavily import TavilySearch

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):

    raise ValueError(
        "TAVILY_API_KEY not found in .env"
    )
    
web_search_tool = TavilySearch(
        max_results=3,
        topic="news",
        search_depth="fast",
        timeout = 180
    )
