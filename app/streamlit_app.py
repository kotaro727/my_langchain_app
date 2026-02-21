import streamlit as st
import sys
import os

# Ensure the parent directory is in sys.path so we can import app.agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.debate import build_debate_graph

st.set_page_config(
    page_title="議論AIエージェント (LangGraph)",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ 競技ディベートAIエージェント")
st.markdown("""
LangGraphを使用して構築されたAIディベートシステムです。  
テーマを設定すると、**リサーチャー**が事実を集め、**肯定側**と**否定側**のAIエージェントが自動で白熱したディベートを行います。最後に**審判**が論理的な判定を下します。
""")

# ==========================================
# Sidebar: Settings
# ==========================================
st.sidebar.header("🎯 ディベート設定")
topic = st.sidebar.text_area(
    "議題を入力してください",
    value="AIは人類の雇用を奪うか、それとも豊かにするか",
    height=100
)

max_turns = st.sidebar.slider(
    "ターン数 (肯定・否定の往復回数)",
    min_value=1,
    max_value=5,
    value=3
)

start_button = st.sidebar.button("ディベート開始！", type="primary")

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
            graph = build_debate_graph()
            
            initial_state = {
                "topic": topic,
                "messages": [],
                "research_data": "",
                "turn_count": 0,
                "max_turns": max_turns,
                "judge_result": ""
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
                                # Remove Prefix for cleaner UI
                                clean_msg = msg_content.replace("【肯定側】\n", "")
                                st.markdown(f"**[肯定側]**\n{clean_msg}")
                                
                        elif node_name == "con_debater":
                            with st.chat_message("assistant", avatar="🔴"):
                                msg_content = state_update["messages"][0].content
                                # Remove Prefix for cleaner UI
                                clean_msg = msg_content.replace("【否定側】\n", "")
                                st.markdown(f"**[否定側]**\n{clean_msg}")
                                
                        elif node_name == "judge":
                            st.divider()
                            st.subheader("⚖️ 審判のジャッジとフィードバック")
                            with st.chat_message("assistant", avatar="⚖️"):
                                st.markdown(state_update["judge_result"])
                                
            st.success("ディベートが終了しました。")
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.error("環境変数（OPENAI_API_KEY, TAVILY_API_KEY）が正しく設定されているか確認してください。")
