import operator
import os
from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# 1. 状態（State）の定義
# ==========================================
class DebateState(TypedDict):
    topic: str
    messages: Annotated[List[BaseMessage], add_messages]
    research_data: str
    turn_count: int
    max_turns: int
    judge_result: str

# ==========================================
# 2. ノード（エージェント）の定義
# ==========================================

def researcher_node(state: DebateState):
    """検索を使って議題に関する事実情報を収集します。"""
    topic = state["topic"]
    
    # 検索ツールを初期化（環境変数にTAVILY_API_KEYが設定されていることを前提とします）
    search_tool = TavilySearchResults(max_results=3)
    
    try:
        search_query = f"{topic} 事実 背景 メリット デメリット"
        search_results = search_tool.invoke({"query": search_query})
        formatted_data = f"【リサーチ結果】\n{search_results}"
    except Exception as e:
        formatted_data = f"【リサーチ結果】\nリサーチ情報の取得に失敗しました: {e}"
        
    return {"research_data": formatted_data}


def pro_debater_node(state: DebateState):
    """肯定側のディベーターとして振る舞います。"""
    # ディベートの多様な切り口や創造的な反論を引き出すため、temperatureを少し高め(0.7)に設定
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    system_prompt = f"""あなたは競技ディベートの「肯定側（賛成派）」エージェントです。
以下の議題について、肯定側の立場から強力で論理的な主張や、否定側への反論を行ってください。

【議題】: {state['topic']}

【リサーチデータ】（客観的事実として活用してください）:
{state['research_data']}

簡潔かつ説得力のある論理を展開してください。
"""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    # 発言者が誰かを明確にするためにプレフィックスを追加
    pro_message = AIMessage(content=f"【肯定側】\n{response.content}")
    
    return {"messages": [pro_message]}


def con_debater_node(state: DebateState):
    """否定側のディベーターとして振る舞います。"""
    # 肯定側と同様に、多様な反論や創造的な主張を引き出すためtemperature=0.7に設定
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    system_prompt = f"""あなたは競技ディベートの「否定側（反対派）」エージェントです。
以下の議題について、否定側の立場から強力で論理的な主張や、肯定側への反論を行ってください。

【議題】: {state['topic']}

【リサーチデータ】（客観的事実として活用してください）:
{state['research_data']}

簡潔かつ説得力のある論理を展開し、一つ前の肯定側の主張に対する反論を必ず含めてください。
"""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    con_message = AIMessage(content=f"【否定側】\n{response.content}")
    
    return {
        "messages": [con_message],
        "turn_count": state.get("turn_count", 0) + 1  # 否定側の終了後にターン数をインクリメント
    }


def judge_node(state: DebateState):
    """ディベートを評価する審判として振る舞います。"""
    # 審判は客観的かつ一貫性のある論理的な評価が求められるため、temperatureを低め(0.2)に設定して出力を安定させる
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    system_prompt = f"""あなたは競技ディベートの「審判（ジャッジ）」エージェントです。
以下の議題についての肯定側と否定側のディベート内容を読み、どちらの論理がより妥当であったかを客観的に判定してください。

【議題】: {state['topic']}

判定の際は、以下の点を含めて詳細なフィードバック（判断理由）を出力してください：
1. 勝敗（どちらがより説得力があったか）
2. 肯定側の良かった点、論理の穴
3. 否定側の良かった点、論理の穴
4. ユーザーが論理的思考力を高めるためのアドバイス

ユーザーの学びになるように、なぜその反論が有効だったのか、どこに詭弁や論理の飛躍があったのかを明示的に解説してください。
"""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    
    return {"judge_result": response.content}


# ==========================================
# 3. グラフのルーティングとコンパイル
# ==========================================

def should_continue(state: DebateState) -> Literal["pro_debater", "judge"]:
    """ディベートを続けるか、審判に移行するかを決定します。"""
    current_turn = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 3)
    
    if current_turn >= max_turns:
        return "judge"
    return "pro_debater"


def build_debate_graph():
    """ディベートエージェント用のStateGraphを構築し、コンパイルします。"""
    workflow = StateGraph(DebateState)
    
    # ノードの追加
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("pro_debater", pro_debater_node)
    workflow.add_node("con_debater", con_debater_node)
    workflow.add_node("judge", judge_node)
    
    # エントリポイント（開始地点）の設定
    workflow.set_entry_point("researcher")
    
    # 通常のエッジ（つながり）の追加
    workflow.add_edge("researcher", "pro_debater")
    workflow.add_edge("pro_debater", "con_debater")
    
    # 否定側のターンの後に条件分岐エッジを追加
    workflow.add_conditional_edges(
        "con_debater",
        should_continue,
        {
            "pro_debater": "pro_debater",
            "judge": "judge"
        }
    )
    
    # 審判の後に終了エッジを追加
    workflow.add_edge("judge", END)
    
    return workflow.compile()

# テスト実行用のヘルパー関数
def run_debate(topic: str, max_turns: int = 3):
    """ディベートを実行するためのヘルパー関数です。"""
    graph = build_debate_graph()
    
    initial_state = {
        "topic": topic,
        "messages": [],
        "research_data": "",
        "turn_count": 0,
        "max_turns": max_turns,
        "judge_result": ""
    }
    
    print(f"=== ディベート開始: {topic} ===")
    print(f"(設定ターン数: {max_turns}ターン)\n")
    
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in event.items():
            if node_name == "researcher":
                print("🔍 [リサーチャー]: 背景情報を収集しました。\n")
            elif node_name == "pro_debater":
                print(state_update["messages"][0].content)
                print("-" * 50)
            elif node_name == "con_debater":
                print(state_update["messages"][0].content)
                print("-" * 50)
            elif node_name == "judge":
                print("⚖️ [審判のジャッジとフィードバック]:\n")
                print(state_update["judge_result"])
                
if __name__ == "__main__":
    # テスト実行
    run_debate("AIは人類の雇用を奪うか、それとも豊かにするか")
