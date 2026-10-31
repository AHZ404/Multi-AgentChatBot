import streamlit as st
from router import route_query

st.set_page_config(page_title="Support Bot", layout="wide")
st.title("🤖 Multi-Agent Support Bot")

with st.sidebar:
    st.subheader("Routing Insights")
    intent_placeholder = st.empty()
    conf_label_placeholder = st.empty()
    conf_progress_placeholder = st.empty()
    agent_placeholder = st.empty()

if "history" not in st.session_state:
    st.session_state.history = []

# Display previous chat history on app re-run
for turn in st.session_state.history:
    st.chat_message(turn["role"]).write(turn["content"])

if user_msg := st.chat_input("How can I help?"):
    # Render user message
    st.chat_message("user").write(user_msg)

    # Process through backend
    with st.spinner("Classifying intent & routing to agent..."):
        try:
            # Pass past chat history into routing pipeline
            result = route_query(user_msg, st.session_state.history)

            reply = result["response"]
            intent = result["intent"].capitalize()
            confidence = float(max(0.0, min(1.0, result["confidence"])))

            # Render sidebar metrics
            intent_placeholder.metric("Predicted intent", intent)
            conf_label_placeholder.write(f"Confidence: **{confidence * 100:.2f}%**")
            conf_progress_placeholder.progress(confidence)
            agent_placeholder.info(f"Handled by: **{intent} Agent**")

        except Exception as e:
            # Graceful error handling for API failures/rate limits
            reply = "⚠️ An error occurred or API rate limit was hit. Please try again shortly."
            st.error(f"Execution Error: {e}")

    # Render assistant response
    st.chat_message("assistant").write(reply)

    # Save conversation turn to session state history
    st.session_state.history += [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]