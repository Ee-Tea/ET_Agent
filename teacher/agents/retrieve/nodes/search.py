from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from ddgs import DDGS
from .milvus_search import milvus_tool

wiki_tool = Tool(
    name="Wikipedia Search",
    func=WikipediaAPIWrapper(lang="ko").run,
    description="질문에 포함된 단어에 대해 위키백과에서 정보를 검색할 때 사용"
)

def ddg_search(query: str, max_results: int = 5) -> list:
    """
    DuckDuckGo 검색 결과를 가져오는 함수
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, safesearch="off", max_results=max_results):
            results.append({
                "title": r.get("title"),
                "body": r.get("body"),
                "href": r.get("href")
            })
    return results

ddg_tool = Tool(
    name="DuckDuckGo Search",
    description="DuckDuckGo에서 웹 검색 결과를 반환합니다.",
    func=lambda q: ddg_search(q),
)