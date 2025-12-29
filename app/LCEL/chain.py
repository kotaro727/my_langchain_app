from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザが入力した料理のレシピを生成してください。"),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

chain = prompt | llm | parser

# print(chain.invoke({"input": "カレー"}))

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

# Chain of Thoughtと結論をまとめる
cot_and_summary_chain = cot_chain | summary_chain