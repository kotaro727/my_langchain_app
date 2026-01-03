from langchain_community.retrievers import TavilySearchAPIRetriever
from dotenv import load_dotenv
from enum import Enum
from langchain_community.document_loaders import GitLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel
from typing import Any
from langchain_core.documents import Document

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

load_dotenv()

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="langchain==0.2.13",
    file_filter=file_filter,
)

documents = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)
retriever = db.as_retriever()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template('''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問:{question}
''')

# LangChainのドキュメントから検索するretrieverとwebから検索するretrieverを用意する
langchain_retriever = retriever.with_config({"run_name": "langchain_retriever"})
web_retriever = TavilySearchAPIRetriever(k=3).with_config({"run_name": "web_retriever"})

# Routeを用意
class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"

class RouteOutPut(BaseModel):
    route: Route

route_prompt = ChatPromptTemplate.from_template('''\
    以下の質問に回答するための適切なretrieverを選択してください。

    質問: {question}
''')

route_chain = route_prompt | model.with_structured_output(RouteOutPut) | (lambda x: x.route)

# ルーティングの結果を踏まえて検索するretriever
def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]
    if route == Route.langchain_document:
        return langchain_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)
    else:
        raise ValueError(f"Invalid route: {route}")

# RAGチェーンの作成
rag_chain = (
    RunnablePassthrough.assign(route=route_chain)
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke({"question": "東京の天気を教えてください"}))