from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke("LangChainとは何ですか？")

print(response.content)
