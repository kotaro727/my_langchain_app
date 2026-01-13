from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas import RunConfig

from langsmith import Client

load_dotenv()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

# 1. データ読み込み
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="langchain==0.2.13",
    file_filter=file_filter,
)
documents = loader.load()

# ドキュメントを中サイズのチャンクに分割（101-500トークンの範囲）
# これによりHeadlineSplitterを使わないパイプラインが選択される
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # 約375トークン（1トークン≒4文字）
    chunk_overlap=200,
    length_function=len,
)
split_documents = text_splitter.split_documents(documents[:100])

# Ragasが使用するメタデータである「filename」を設定
for doc in split_documents:
    if "filename" not in doc.metadata:
        doc.metadata["filename"] = doc.metadata.get("source", "unknown")

print(f"分割後のドキュメント数: {len(split_documents)}")

# 2. Generator作成
generator = TestsetGenerator.from_langchain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    embedding_model=OpenAIEmbeddings()
)

# 3. query_distributionを作成（各synthesizer と比率のタプルのリスト）
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 0.5),      # 単純な質問 50%
    (MultiHopAbstractQuerySynthesizer(llm=generator.llm), 0.25),      # 抽象的な推論が必要な質問 25%
    (MultiHopSpecificQuerySynthesizer(llm=generator.llm), 0.25),      # 複数の箇所を参照する質問 25%
]

# 4. RunConfigでレート制限を設定
run_config = RunConfig(max_workers=4, max_wait=180)

# 5. テストデータ生成
# 分割したドキュメントを使用
testset = generator.generate_with_langchain_docs(
    split_documents,
    testset_size=4,
    query_distribution=query_distribution,
    run_config=run_config
)

# 結果表示
# to_pandas()メソッドでDataFrameに変換して表示
# 注: IDEの型推論が正しくない場合がありますが、実行時には動作します
df = testset.to_pandas()  # type: ignore
print(df)
print(f"\n生成された質問数: {len(df)}")

# テストセットをLangSmithに保存
client = Client()

dataset_name = "agent-book"

if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(dataset_name=dataset_name)

inputs = []
outputs = []
metadatas = []

# DataFrameから各行を取得してLangSmithのexampleとして追加
for _, row in df.iterrows():
    # Ragasのtestsetには通常以下のカラムが含まれる
    # - user_input: ユーザーの質問
    # - reference/reference_answer: 参照回答
    # - reference_contexts: 参照コンテキスト

    inputs.append({
        "question": row.get("user_input", "")
    })

    outputs.append({
        "answer": row.get("reference", row.get("reference_answer", "")),
        "contexts": row.get("reference_contexts", [])
    })

    metadatas.append({
        "synthesizer_name": row.get("synthesizer_name", "unknown"),
        "episode_done": row.get("episode_done", True)
    })

# LangSmithにexamplesを一括作成
client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadata=metadatas,
    dataset_id=dataset.id
)

print(f"\n✓ LangSmithにデータセット '{dataset_name}' を保存しました")
print(f"  - Examples数: {len(inputs)}")
print(f"  - Dataset ID: {dataset.id}")