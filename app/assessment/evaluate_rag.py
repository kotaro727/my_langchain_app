"""
LangSmithに保存したテストデータセットを使用してRAGシステムをRagasで評価
"""
from dotenv import load_dotenv
from langsmith import Client
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


# --- 1. 準備 ---
print("="*60)
print("RAGシステムの評価を開始")
print("="*60)

client = Client()
dataset_name = "agent-book"  # 01.pyで保存したデータセット名

# --- 2. 評価対象のRAGシステムを構築 ---
print("\n[1] RAGシステムを構築中...")

# ドキュメントの読み込み
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="langchain==0.2.13",
    file_filter=file_filter,
)
documents = loader.load()
print(f"  - 読み込んだドキュメント数: {len(documents)}")

# ベクトルストアの作成
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("  - ベクトルストアを作成完了")

# プロンプトテンプレート
prompt = ChatPromptTemplate.from_template("""
以下の文脈だけを踏まえて質問に回答してください。

文脈:
{context}

質問: {question}
""")

# LLMモデル
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# RAGチェーンの作成
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
    }
    | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"])
    )
    | prompt
    | model
    | StrOutputParser()
)

print("  - RAGチェーンを構築完了")


# --- 3. データの取得とRAGアプリの実行 ---
print(f"\n[2] データセット '{dataset_name}' を取得中...")
examples = list(client.list_examples(dataset_name=dataset_name))
print(f"  - 取得したExample数: {len(examples)}")

data_for_ragas = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

print("\n[3] RAGアプリケーションを実行中...")
for i, example in enumerate(examples, 1):
    question = example.inputs["question"]
    ground_truth = example.outputs.get("answer", "")

    print(f"  [{i}/{len(examples)}] 質問: {question[:50]}...")

    # ★ RAGアプリを実行
    # コンテキストを取得
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    # 回答を生成
    answer = rag_chain.invoke({"question": question})

    # リストに追加
    data_for_ragas["question"].append(question)
    data_for_ragas["answer"].append(answer)
    data_for_ragas["contexts"].append(contexts)
    data_for_ragas["ground_truth"].append(ground_truth)

print("\n  - すべてのデータに対して実行完了")


# --- 4. Ragas用データセットへの変換 ---
print("\n[4] Ragas用データセットに変換中...")
ragas_dataset = Dataset.from_dict(data_for_ragas)
print(f"  - データセットサイズ: {len(ragas_dataset)}")


# --- 5. 評価実行 (採点) ---
print("\n[5] Ragasで評価を実行中...")
print("  使用メトリクス:")
print("    - faithfulness: 回答がコンテキストに忠実か（嘘をついていないか）")
print("    - answer_relevancy: 回答が質問に関連しているか")
print("    - context_precision: 取得したコンテキストの精度")
print("    - context_recall: 必要なコンテキストを取得できたか")

# 審査員モデルは軽量な gpt-4o-mini を推奨
evaluator_llm = ChatOpenAI(model="gpt-4o-mini")

results = evaluate(
    dataset=ragas_dataset,
    metrics=[
        faithfulness,        # 嘘をついていないか
        answer_relevancy,    # 質問に答えているか
        context_precision,   # 検索精度
        context_recall,      # 必要なコンテキストの取得率
    ],
    llm=evaluator_llm,
    embeddings=OpenAIEmbeddings()
)

print("\n  - 評価完了!")


# --- 6. 結果表示 ---
print("\n" + "="*60)
print("評価結果")
print("="*60)

print("\n[全体スコア]")
for metric_name, score in results.items():
    if isinstance(score, (int, float)):
        print(f"  {metric_name}: {score:.3f}")

# Pandasで詳細を表示
df_results = results.to_pandas()

print("\n[詳細結果（最初の3件）]")
# 表示するカラムを選択
display_columns = ["question", "answer", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
available_columns = [col for col in display_columns if col in df_results.columns]

print(df_results[available_columns].head(3).to_string(index=False))

# --- 7. 結果をCSV保存 ---
output_file = "evaluation_result.csv"
df_results.to_csv(output_file, index=False)
print(f"\n✓ 評価結果を '{output_file}' に保存しました")

# --- 8. サマリー統計 ---
print("\n" + "="*60)
print("統計サマリー")
print("="*60)

print("\n[メトリクスの統計]")
for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    if col in df_results.columns:
        print(f"\n{col}:")
        print(f"  - 平均: {df_results[col].mean():.3f}")
        print(f"  - 最小: {df_results[col].min():.3f}")
        print(f"  - 最大: {df_results[col].max():.3f}")
        print(f"  - 標準偏差: {df_results[col].std():.3f}")

# --- 9. 低スコアの質問を抽出 ---
print("\n" + "="*60)
print("改善が必要な質問（低スコア）")
print("="*60)

threshold = 0.5  # しきい値

for metric in ["faithfulness", "answer_relevancy"]:
    if metric in df_results.columns:
        low_score_items = df_results[df_results[metric] < threshold]
        if len(low_score_items) > 0:
            print(f"\n[{metric} < {threshold}]")
            for idx, row in low_score_items.iterrows():
                print(f"  質問: {row['question'][:60]}...")
                print(f"    スコア: {row[metric]:.3f}")
                print(f"    回答: {row['answer'][:80]}...")
                print()

print("\n" + "="*60)
print("評価完了！")
print("="*60)
