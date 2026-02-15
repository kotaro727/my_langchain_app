# ステート: グラフの状態を管理

import operator
from typing import Annotated

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