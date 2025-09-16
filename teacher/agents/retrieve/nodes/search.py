from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from .milvus_search import milvus_tool

# DDG 검색 클라이언트 호환 임포트 (ddgs → duckduckgo_search 순서로 시도)
try:
    from ddgs import DDGS as _DDGS
except Exception:
    try:
        from duckduckgo_search import DDGS as _DDGS
    except Exception:
        _DDGS = None

wiki_tool = Tool(
    name="Wikipedia Search",
    func=WikipediaAPIWrapper(lang="ko").run,
    description="질문에 포함된 단어에 대해 위키백과에서 정보를 검색할 때 사용"
)

def ddg_search(query: str, max_results: int = 5) -> list:
    """
    DuckDuckGo 검색 결과를 가져오는 함수
    """
    if _DDGS is None:
        return []
    results = []
    with _DDGS() as ddgs:
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