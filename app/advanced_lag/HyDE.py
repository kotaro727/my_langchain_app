from langchain_community.document_loaders import GitLoader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="langchain==0.2.13",
    file_filter=file_filter,
)

documents = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template('''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問:{question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

# LLMに回答の仮説を作らせる
hypothetical_prompt = ChatPromptTemplate.from_template('''\
    次の質問に回答する一文を書いてください。

    質問:{question}
''')

# 仮説を作成するChain
hyde_chain = hypothetical_prompt | model | StrOutputParser()

hyde_rag_chain = {
    "question": RunnablePassthrough(),
    "context": hyde_chain | retriever, # hyde_chainで仮説を作成したら、それをRetrieverに渡す
} | prompt | model | StrOutputParser()

print(hyde_rag_chain.invoke({"question": "LangChainの概要を教えて"}))
