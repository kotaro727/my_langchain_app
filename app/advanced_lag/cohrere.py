from langchain_community.document_loaders import GitLoader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

load_dotenv()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

# ドキュメントのロード
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="langchain==0.2.13",
    file_filter=file_filter,
)

documents = loader.load()

# print(documents)
# print(len(documents))

# Embeddingsとベクトルストアの設定
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)

# 基本的なRetrieverの作成
base_retriever = db.as_retriever(search_kwargs={"k": 10})

# Cohere Rerankの設定
# top_nパラメータで最終的に返すドキュメント数を指定
cohere_rerank = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5
)

# ContextualCompressionRetrieverを使用してリランクを適用
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_rerank,
    base_retriever=base_retriever
)

# RAGチェーンのためのプロンプトテンプレート
prompt = ChatPromptTemplate.from_template('''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈: """
    {context}
    """

    質問:{question}
''')

# LLMモデルの設定
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Cohere Rerankを使用したRAGチェーン
rerank_chain = {
    "question": RunnablePassthrough(),
    "context": RunnableLambda(itemgetter("question")) | compression_retriever,
} | prompt | model | StrOutputParser()

# チェーンの実行
print(rerank_chain.invoke({"question": "LangChainとは何ですか？"}))

