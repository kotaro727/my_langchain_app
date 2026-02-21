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
# 1. 状態（State）の定義 (Reflection用に追加)
# ==========================================
class DebateReflectionState(TypedDict):
    topic: str
    messages: Annotated[List[BaseMessage], add_messages]
    research_data: str
    round_count: int      # ターン数ではなく「ラウンド数」を管理
    max_rounds: int       # 最大ラウンド数
    feedback: str         # 中間ジャッジからのフィードバック
    final_judge_result: str

# ==========================================
# 2. ノード（エージェント）の定義
# ==========================================

def researcher_node(state: DebateReflectionState):
    """検索を使って議題に関する事実情報を収集します。"""
    topic = state["topic"]
    search_tool = TavilySearchResults(max_results=3)
    
    try:
        search_query = f"{topic} 事実 背景 メリット デメリット"
        search_results = search_tool.invoke({"query": search_query})
        
        # LLMが事実を誤認しないよう、辞書のリストから綺麗なテキストに整形する
        formatted_data = "【リサーチ結果】\n"
        for i, res in enumerate(search_results):
            title = res.get('title', 'No Title')
            url = res.get('url', 'No URL')
            content = res.get('content', '')
            formatted_data += f"\n[出典 {i+1}] {title}\nURL: {url}\n内容要約:\n{content}\n"
            
    except Exception as e:
        formatted_data = f"【リサーチ結果】\nリサーチ情報の取得に失敗しました: {e}"
        
    return {"research_data": formatted_data}


def pro_debater_node(state: DebateReflectionState):
    """肯定側のディベーターとして振る舞います。"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    system_prompt = f"""あなたは競技ディベートの「肯定側（賛成派）」エージェントです。
以下の議題について、肯定側の立場から強力で論理的な主張を行ってください。

【議題】: {state['topic']}

【リサーチデータ】:
{state['research_data']}

【重要な指示】
1. もし直前に「否定側」からの反論がある場合は、必ずその否定側の主張の弱点や論理の穴を指摘し、直接的に再反論してください。
2. 簡潔かつ説得力のある論理を展開してください。
3. **【ハルシネーション（捏造）の絶対禁止】** 主張の根拠は、必ず上記の【リサーチデータ】に記載されている事実のみを使用してください。架空のデータや事例を作り出すことは厳禁です。
4. **【出典の明記】** リサーチデータの内容を引用・参照する場合は、必ず文中に `[出典 1]` のように参照元を明記してください。
"""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # ★ Reflectionパターンのキモ: フィードバックはシステムプロンプトではなく、履歴の「最後」にHumanMessageとして追加する（LLMの指示無視を防ぐため）
    if state.get("feedback"):
        feedback_prompt = f"【中間ジャッジからのフィードバック（改善要求）】\n{state['feedback']}\n\n※上記のフィードバックのうち、**「肯定側へのフィードバック」**を特に深く受け止め、自分の論理の弱点を補強・修正した上で再主張（および相手の直前の発言への再反論）を展開してください。肯定側としての役割に徹してください。"
        messages.append(HumanMessage(content=feedback_prompt))
    
    response = llm.invoke(messages)
    pro_message = AIMessage(content=f"【肯定側】\n{response.content}")
    
    return {"messages": [pro_message]}


def con_debater_node(state: DebateReflectionState):
    """否定側のディベーターとして振る舞います。"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    system_prompt = f"""あなたは競技ディベートの「否定側（反対派）」エージェントです。
以下の議題について、否定側の立場から強力で論理的な主張や、肯定側への反論を行ってください。

【議題】: {state['topic']}

【リサーチデータ】:
{state['research_data']}

簡潔かつ説得力のある論理を展開し、一つ前の肯定側の主張に対する反論を必ず含めてください。

【重要な指示】
1. **【ハルシネーション（捏造）の絶対禁止】** 主張の根拠は、必ず上記の【リサーチデータ】に記載されている事実のみを使用してください。架空のデータや事例を作り出すことは厳禁です。
2. **【出典の明記】** リサーチデータの内容を引用・参照する場合は、必ず文中に `[出典 1]` のように参照元を明記してください。
"""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # ★ 否定側も最新の指示としてフィードバックを末尾のHumanMessageとして追加する
    if state.get("feedback"):
        feedback_prompt = f"【中間ジャッジからのフィードバック（改善要求）】\n{state['feedback']}\n\n※上記のフィードバックのうち、**「否定側へのフィードバック」**を特に深く受け止め、自分の論理の弱点を補強・修正した上で相手の直前の発言への再反論を展開してください。否定側としての役割に徹してください。"
        messages.append(HumanMessage(content=feedback_prompt))
    
    response = llm.invoke(messages)
    con_message = AIMessage(content=f"【否定側】\n{response.content}")
    
    return {"messages": [con_message]}


def intermediate_judge_node(state: DebateReflectionState):
    """ラウンド終了時に両者の論理をレビューし、改善のためのフィードバック（ダメ出し）を行います。"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    system_prompt = f"""あなたは競技ディベートの「中間レビュアー（コーチ）」です。
現在の議題「{state['topic']}」について、肯定側と否定側のやり取りを分析してください。

【リサーチデータ（ファクトチェック用）】:
{state['research_data']}

【あなたの役割】
勝敗を決めるのではなく、**両者の議論をより深く、ファクト（事実）に基づいた建設的なものにするための厳しいダメ出し（フィードバック）**を行ってください。
以下の点を厳しくチェックし、次のラウンドで両者が何を改善すべきかを明確に指示してください。

1. **ファクトチェック（捏造の摘発）**: 両者の主張が【リサーチデータ】に基づいているか確認してください。もしデータにない架空の数値や事例（ハルシネーション）を語っている場合は、「〇〇というデータはリサーチ結果に存在しません」と厳しく指摘してください。
2. **論理の飛躍と証拠不足**: データや具体例が不足している点、相手の反論にまともに答えていない点を指摘してください。
"""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "feedback": response.content,
        "round_count": state.get("round_count", 0) + 1 # ラウンド数をインクリメント
    }


def final_judge_node(state: DebateReflectionState):
    """指定ラウンド終了後に、最終的な勝敗とフィードバックを下す審判です。"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    system_prompt = f"""あなたは競技ディベートの「最終審判」エージェントです。
議題「{state['topic']}」についてのこれまでの全議論（フィードバックとそれに応じた改善プロセスを含む）を読み、最終的な勝敗を客観的に判定してください。

以下の点を含めて詳細なフィードバックを出力してください：
1. 最終的な勝敗と、その決定打となった理由
2. 各陣営がフィードバックを受けてどのように議論を改善できたか（またはできなかったか）
3. ユーザーが論理的思考力を高めるための総括アドバイス
"""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    
    return {"final_judge_result": response.content}


# ==========================================
# 3. グラフのルーティングとコンパイル
# ==========================================

def should_continue(state: DebateReflectionState) -> Literal["pro_debater", "final_judge"]:
    """指定ラウンド数に達したか確認し、ループするか最終判断に進むか決定します。"""
    current_round = state.get("round_count", 0)
    max_rounds = state.get("max_rounds", 2)
    
    if current_round >= max_rounds:
        return "final_judge"
    return "pro_debater"


def build_reflection_debate_graph():
    """Reflectionパターンを組み込んだStateGraphを構築します。"""
    workflow = StateGraph(DebateReflectionState)
    
    # ノードの追加
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("pro_debater", pro_debater_node)
    workflow.add_node("con_debater", con_debater_node)
    workflow.add_node("intermediate_judge", intermediate_judge_node) # 新規追加
    workflow.add_node("final_judge", final_judge_node)             # 新規追加
    
    # 通常のエッジ（Start -> プロ -> コン -> 中間ジャッジ）
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "pro_debater")
    workflow.add_edge("pro_debater", "con_debater")
    workflow.add_edge("con_debater", "intermediate_judge") # 否定側の後は中間ジャッジへ
    
    # 条件分岐: 中間ジャッジのフィードバック後、プロに戻るか最終ジャッジへ行くか
    workflow.add_conditional_edges(
        "intermediate_judge",
        should_continue,
        {
            "pro_debater": "pro_debater",
            "final_judge": "final_judge"
        }
    )
    
    workflow.add_edge("final_judge", END)
    
    return workflow.compile()

# ==========================================
# 4. 実行用クラス/関数
# ==========================================
def run_reflection_debate(topic: str, max_rounds: int = 2):
    """ターミナル上でReflectionディベートを実行する関数"""
    graph = build_reflection_debate_graph()
    
    initial_state = {
        "topic": topic,
        "messages": [],
        "research_data": "",
        "round_count": 0,
        "max_rounds": max_rounds,
        "feedback": "",
        "final_judge_result": ""
    }
    
    print(f"=== 🔄 Reflection ディベート開始: {topic} ===")
    print(f"(設定ラウンド数: {max_rounds}ラウンド)\n")
    
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
            elif node_name == "intermediate_judge":
                print("💡 [中間ジャッジからのダメ出し（Reflection）]:")
                print(state_update["feedback"])
                print("=" * 50)
            elif node_name == "final_judge":
                print("⚖️ [最終審判のジャッジと総括]:\n")
                print(state_update["final_judge_result"])
                
if __name__ == "__main__":
    # お試し実行（2ラウンド）
    run_reflection_debate("日本は週休3日制を法制化すべきか", max_rounds=2)
