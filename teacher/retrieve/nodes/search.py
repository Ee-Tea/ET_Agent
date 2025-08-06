from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun

wiki_tool = Tool(
    name="Wikipedia Search",
    func=WikipediaAPIWrapper(lang="ko").run,
    description="질문에 포함된 단어에 대해 위키백과에서 정보를 검색할 때 사용"
)

ddg_search = DuckDuckGoSearchRun()

ddg_tool = Tool(
    name="DuckDuckGo Search",
    func=ddg_search.run,
    description="질문에 포함된 단어나 문장에 대해 DuckDuckGo에서 정보를 검색할 때 사용"
)