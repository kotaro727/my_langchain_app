"""
LangServeサーバー
レシピ生成チェーンをREST APIとして公開します。
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv

from app.LCEL.chain_for_serve import chain

load_dotenv()

app = FastAPI(
    title="レシピ生成API",
    version="1.0.0",
    description="ユーザーが入力した料理名からレシピを生成するAPI",
)

# チェーンをエンドポイントとして追加
add_routes(
    app,
    chain,
    path="/recipe",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

