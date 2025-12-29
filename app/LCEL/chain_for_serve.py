"""
LangServe用のチェーン定義
レシピ生成チェーンをLangServeで使用するためのモジュール
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# レシピ生成用のプロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザが入力した料理のレシピを生成してください。"),
    ("human", "{input}"),
])

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 出力パーサー
parser = StrOutputParser()

# LangServeで使用するチェーン
chain = prompt | llm | parser

