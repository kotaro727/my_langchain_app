from langchain_community.document_loaders import GitLoader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

class MultipleQueries(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="langchain==0.2.13",
    file_filter=file_filter,
)

documents = loader.load()

# print(documents)
# print(len(documents))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)
retriever = db.as_retriever()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# RunnableParallelを使用したChain
prompt = ChatPromptTemplate.from_template('''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問:{question}
''')

query_geration_prompt = ChatPromptTemplate.from_template('''\
    以下の質問に対して、5つの異なる検索クエリを生成してください。

    質問: {question}
''')

query_generation_chain = query_geration_prompt | model.with_structured_output(MultipleQueries) | (lambda x: x.queries)

multiple_queries_rag_chain = {
    "question": RunnablePassthrough(),
    "context": query_generation_chain | retriever.map()
} | prompt | model | StrOutputParser()

print(multiple_queries_rag_chain.invoke({"question": "LangChainとは何ですか？"}))