# ==========================================
# 開発環境の立ち上げ方
# ==========================================
# 1. プロジェクトルートディレクトリで仮想環境をアクティベートする（macOS/Linuxの場合）
#    source venv/bin/activate
#
# 2. 必要な環境変数（OPENAI_API_KEY, TAVILY_API_KEY）が .env などに設定されていることを確認する
#
# 3. Streamlitアプリを起動する（以下のコマンドを実行）
#    streamlit run app/agent/streamlit_reflection_app.py
# ==========================================

import streamlit as st
import sys
import os

# appモジュールなどを正しくimportできるようにプロジェクトルートをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agent.debate_reflection import build_reflection_debate_graph

st.set_page_config(
    page_title="Reflection議論AIエージェント (LangGraph)",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Reflection競技ディベートAIエージェント")
st.markdown("""
LangGraphを使用して構築された、**Reflection（自己内省）パターン**を組み込んだAIディベートシステムです。  
**「肯定側 ➡️ 否定側 ➡️ 中間ジャッジによるダメ出し」** を1ラウンドとし、フィードバックを受けて議論を深化させていきます。
""")

# ==========================================
# Sidebar: Settings
# ==========================================
st.sidebar.header("🎯 ディベート設定")
topic = st.sidebar.text_area(
    "議題を入力してください",
    value="日本は週休3日制を法制化すべきか",
    height=100
)

max_rounds = st.sidebar.slider(
    "最大ラウンド数 (ダメ出しを受けて再議論する回数)",
    min_value=1,
    max_value=3,
    value=2
)

start_button = st.sidebar.button("Reflectionディベート開始！", type="primary")

# ==========================================
# Main UI Logic
# ==========================================
if start_button:
    if not topic.strip():
        st.warning("議題を入力してください。")
    else:
        st.write(f"### 議題: **{topic}**")
        st.info("ディベートを構築・実行しています... 少々お待ちください。")
        
        # UI Container for chat messages
        chat_container = st.container()
        
        try:
            # Graph Initialization
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
            
            # Stream events and update UI in real-time
            with chat_container:
                for event in graph.stream(initial_state, stream_mode="updates"):
                    for node_name, state_update in event.items():
                        
                        if node_name == "researcher":
                            with st.chat_message("assistant", avatar="🔍"):
                                st.markdown("**[リサーチャー]**\n背景情報・事実データを収集しました。")
                                with st.expander("リサーチデータを見る"):
                                    st.markdown(state_update["research_data"])
                        
                        elif node_name == "pro_debater":
                            with st.chat_message("user", avatar="🔵"):
                                msg_content = state_update["messages"][0].content
                                # Replace Prefix for cleaner UI if present
                                clean_msg = msg_content.replace("【肯定側】\n", "")
                                st.markdown(f"**[肯定側]**\n{clean_msg}")
                                
                        elif node_name == "con_debater":
                            with st.chat_message("assistant", avatar="🔴"):
                                msg_content = state_update["messages"][0].content
                                # Replace Prefix for cleaner UI if present
                                clean_msg = msg_content.replace("【否定側】\n", "")
                                st.markdown(f"**[否定側]**\n{clean_msg}")
                                
                        elif node_name == "intermediate_judge":
                            with st.chat_message("assistant", avatar="💡"):
                                current_round = state_update.get("round_count", "?")
                                st.markdown(f"**[中間ジャッジからのダメ出し（第 {current_round} ラウンド修了）]**\n\n{state_update['feedback']}")
                                
                        elif node_name == "final_judge":
                            st.divider()
                            st.subheader("⚖️ 最終審判のジャッジと総括")
                            with st.chat_message("assistant", avatar="⚖️"):
                                st.markdown(state_update["final_judge_result"])
                                
            st.success("Reflectionディベートが終了しました。")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.error("環境変数（OPENAI_API_KEY, TAVILY_API_KEY）が正しく設定されているか確認してください。")
