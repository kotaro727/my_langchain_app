# LCEL（LangChain Expression Language）ガイド

## 概要

**LCEL（LangChain Expression Language）**は、LangChain で LLM アプリケーションを構築するための宣言的な記法です。`|`演算子を使ってコンポーネントを連結することで、複雑な処理フローを簡潔に記述できます。

## 主な特徴

### 1. 宣言的な記法

`|`演算子を使ってコンポーネントを連結することで、処理の流れを直感的に表現できます。

```python
chain = prompt | llm | parser
```

### 2. 自動サポート機能

LCEL で作成したチェーンは、以下の機能を自動的にサポートします：

- **同期実行**（`invoke`）
- **非同期実行**（`ainvoke`）
- **バッチ処理**（`batch`/`abatch`）
- **ストリーミング**（`stream`/`astream`）

### 3. 型安全性

各ステップの入出力型が自動的に推論され、型チェックが効きます。

## 基本的な使い方

### シンプルなチェーン

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# プロンプトテンプレート
prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザが入力した料理のレシピを生成してください。"),
    ("human", "{input}"),
])

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 出力パーサー
parser = StrOutputParser()

# LCELでチェーンを構築
chain = prompt | llm | parser

# 実行
result = chain.invoke({"input": "カレー"})
```

### 複雑なチェーンの構築

複数のチェーンを組み合わせて、より複雑な処理フローを構築できます：

```python
# Chain of Thoughtで回答を生成
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問にステップバイステップで回答してください。"),
    ("human", "{question}"),
])

cot_chain = cot_prompt | llm | parser

# COTで生成した回答から結論を抽出
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "ステップバイステップで考えた結論を抽出してください。"),
    ("human", "{text}"),
])

summary_chain = summary_prompt | llm | parser

# チェーンを連結
cot_and_summary_chain = cot_chain | summary_chain

result = cot_and_summary_chain.invoke({"question": "10 + 2 * 3"})
```

## 実行方法

### 1. 同期実行

```python
result = chain.invoke({"input": "カレー"})
```

### 2. 非同期実行

```python
result = await chain.ainvoke({"input": "カレー"})
```

### 3. バッチ処理

```python
results = chain.batch([
    {"input": "カレー"},
    {"input": "パスタ"},
    {"input": "ラーメン"}
])
```

### 4. ストリーミング

```python
for chunk in chain.stream({"input": "カレー"}):
    print(chunk, end="", flush=True)
```

## RunnableSequence について

LCEL の`|`演算子で連結されたチェーンは、内部的に`RunnableSequence`として実装されます。

### RunnableSequence の特徴

- **順次実行**: 前の`Runnable`の出力が次の`Runnable`の入力になる
- **自動型推論**: 各ステップの入出力型が自動的に推論される
- **ストリーミング対応**: すべてのコンポーネントが`transform`を実装している場合、ストリーミングが可能

### 作成方法

```python
# |演算子を使用（推奨）
chain = prompt | llm | parser

# 直接インスタンス化（通常は不要）
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(first=prompt, middle=[], last=parser)
```

## LCEL の利点

1. **可読性**: 処理の流れが直感的に理解できる
2. **再利用性**: コンポーネントを組み合わせて再利用しやすい
3. **拡張性**: 新しいコンポーネントを簡単に追加できる
4. **デバッグ**: 各ステップを個別にテストできる
5. **本番対応**: 非同期、バッチ、ストリーミングなどの機能が自動的にサポートされる

## 実装例

このディレクトリの`chain.py`では、以下のような実装例があります：

```python
# 基本的なレシピ生成チェーン
chain = prompt | llm | parser

# Chain of Thoughtと結論抽出を組み合わせたチェーン
cot_and_summary_chain = cot_chain | summary_chain
```

これらのチェーンは、LCEL の`|`演算子で構築され、`RunnableSequence`として動作します。

## LangSmith との連携

LCEL で作成したチェーンは、LangSmith で自動的にトレーシングされます。環境変数を設定することで、実行ログを LangSmith で確認できます。

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-api-key
LANGCHAIN_PROJECT=my-project
```

## まとめ

LCEL は、LLM アプリケーションを簡潔かつ保守しやすく構築するための強力なツールです。`|`演算子を使ってコンポーネントを連結するだけで、本番レベルの機能（非同期、バッチ、ストリーミング）を自動的に利用できます。

## 参考リンク

- [LangChain 公式ドキュメント](https://python.langchain.com/)
- [LCEL 公式ドキュメント](https://python.langchain.com/docs/expression_language/)
- [LangSmith](https://smith.langchain.com/)
