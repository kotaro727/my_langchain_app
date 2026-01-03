# BM25とReciprocal Rank Fusionを使用したハイブリッド検索
from langchain_community.document_loaders import GitLoader
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from typing import List
import numpy as np

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

# Embeddingsとベクトルストアの設定（ベクトル検索用）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)
vector_retriever = db.as_retriever(search_kwargs={"k": 10})

# BM25検索の設定
# ドキュメントをトークン化（簡易的に空白で分割）
tokenized_docs = [doc.page_content.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

def bm25_retriever(query: str, k: int = 10) -> List[Document]:
    """BM25を使用してドキュメントを検索する"""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # スコアの高い順にインデックスを取得
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    # 該当するドキュメントを返す
    return [documents[i] for i in top_k_indices]

def reciprocal_rank_fusion(results: list[list], k: int = 60) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) アルゴリズム
    複数の検索結果を統合してリランキングする
    
    Args:
        results: 検索結果のリスト（各要素はDocumentのリスト）
        k: RRFの定数パラメータ（デフォルト60）
    
    Returns:
        統合されたドキュメントのリスト（スコアの高い順）
    """
    fused_scores = {}
    doc_map = {}  # ドキュメントの内容からDocumentオブジェクトへのマッピング
    
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            # ドキュメントオブジェクトを保存
            if doc_str not in doc_map:
                doc_map[doc_str] = doc
            
            # RRFスコアを計算: 1 / (rank + k)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    # スコアでソートして上位のドキュメントを返す
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_str] for doc_str, score in reranked_results]

def hybrid_retriever(query: str, k: int = 5, rrf_k: int = 60) -> List[Document]:
    """
    ベクトル検索とBM25検索のハイブリッド検索（RRF使用）
    
    Args:
        query: 検索クエリ
        k: 最終的に返すドキュメント数
        rrf_k: Reciprocal Rank FusionのKパラメータ（デフォルト60）
    """
    # ベクトル検索の結果を取得
    vector_docs = vector_retriever.invoke(query)
    
    # BM25検索の結果を取得
    bm25_docs = bm25_retriever(query, k=10)
    
    # Reciprocal Rank Fusionで2つの検索結果を統合
    fused_docs = reciprocal_rank_fusion([vector_docs, bm25_docs], k=rrf_k)
    
    # 上位k件を返す
    return fused_docs[:k]

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

# ハイブリッド検索を使用したRAGチェーン
hybrid_chain = {
    "question": RunnablePassthrough(),
    "context": RunnableLambda(lambda x: hybrid_retriever(itemgetter("question")(x))),
} | prompt | model | StrOutputParser()

# チェーンの実行
print("=== ハイブリッド検索（BM25 + ベクトル検索 with RRF）の結果 ===")
print(hybrid_chain.invoke({"question": "LangChainとは何ですか？"}))

