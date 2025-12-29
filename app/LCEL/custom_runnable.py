from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

# Custom Runnable
def uppercase(text: str) -> str:
    return text.upper()

# Custom Runnableを使用したChain
# RunnableLambdaは、省略可能
chain = prompt | llm | parser | RunnableLambda(uppercase)

print(chain.invoke({"input": "hello"}))