# LangServe レシピ生成API

このディレクトリには、LangChainのチェーンをLangServeでREST APIとして公開する実装が含まれています。

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルに以下の環境変数を設定してください：

```env
OPENAI_API_KEY=your-openai-api-key
```

## 使い方

### サーバーの起動

```bash
# プロジェクトルートから実行
python -m app.LCEL.serve

# または
cd app/LCEL
python serve.py
```

サーバーは `http://localhost:8000` で起動します。

### APIエンドポイント

#### レシピ生成

**エンドポイント**: `POST /recipe/invoke`

**リクエストボディ**:
```json
{
  "input": "カレー"
}
```

**レスポンス**:
```json
{
  "output": "カレーのレシピ..."
}
```

### APIドキュメント

サーバー起動後、以下のURLでAPIドキュメントを確認できます：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 使用例

#### cURL

```bash
curl -X POST "http://localhost:8000/recipe/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": "カレー"}'
```

#### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/recipe/invoke",
    json={"input": "カレー"}
)
print(response.json())
```

## ファイル構成

- `chain.py`: レシピ生成チェーンの定義
- `serve.py`: LangServeサーバーの実装
- `custom_runnable.py`: カスタムRunnableの例

## 注意事項

- サーバーを起動する前に、`.env`ファイルに`OPENAI_API_KEY`が設定されていることを確認してください
- プロジェクトルートから実行する場合は、`PYTHONPATH`が正しく設定されている必要があります

