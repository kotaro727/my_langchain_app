# ステート: グラフの状態を管理

import operator
from typing import Annotated, Any

from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph

# BaseModelを継承
class State(BaseModel):
    query: str = Field(
        ..., description="ユーザーからの質問"
    )
    # 上書きされる
    current_role: str = Field(
        default="", description="選定された回答ロール（今、誰として答えているか）"
    )
    # add演算子を使って、要素を追加する
    messages: Annotated[list[str], operator.add] = Field(
        default=[], description="回答履歴"
    )
    current_judge: bool = Field(
        default=False, description="品質チェックの結果"
    )
    judgement_reason: str = Field(
        default="", description="品質チェックの判定理由"
    )

workflow = StateGraph(State)

# ノード: グラフを構成する処理の単位
# stateを受け取り、回答を生成して、stateの更新差分を返す
def answering_node(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role

    generated_message = # LLMで回答を生成

    # stateの更新差分を返す
    # compile時にLangGraphがstate.messagesに追加してくれる
    return {
        "messages": [generated_message]
    }

def check_node(state: State) -> dict[str, Any]:
    messages = state.messages
    last_message = messages[-1]

    # 品質チェック
    judge = # 判定結果
    reason = # 判定理由

    return {
        "current_judge": judge,
        "judgement_reason": reason
    }

workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

# エントリーポイント
# set_entry_pointメソッドでエントリーポイントを文字列で設定
workflow.set_entry_point("answering")

# エッジ: ノードとノードをどう繋ぐかのルール
# add_edgeメソッドでエッジを設定。第一引数: 遷移元ノード, 第二引数: 遷移先ノード
workflow.add_edge("answering", "check")

# 条件付きエッジ: 条件付きでエッジを設定
# add_conditional_edgesメソッドでエッジを設定。第一引数: 遷移元ノード, 第二引数: 条件, 第三引数: 第二引数の結果に対応する遷移先ノードとの辞書
workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,
    {
        True: "end",
        False: "answering"
    }
)