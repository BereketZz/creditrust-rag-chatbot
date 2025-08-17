import gradio as gr
from src.rag_pipeline import generate_answer

# ----------------------------
# RAG Chat Interface Functions
# ----------------------------

def rag_chatbot_interface(query, history):
    if not query.strip():
        return history + [("Please enter a valid question.", "")]
    try:
        answer, sources = generate_answer(query)
    except Exception as e:
        return history + [(query, f"⚠️ Error: {str(e)}")]

    # Build formatted sources
    source_display = "\n\n".join(
        [f"📄 Source {i+1}:\n{chunk[:300]}..." for i, chunk in enumerate(sources[:3])]
    )

    # Append answer + sources to chat history
    history.append((query, f"💡 {answer}\n\n---\n📚 **Top Sources**:\n{source_display}"))
    return history

# ----------------------------
# Gradio Interface Layout
# ----------------------------

with gr.Blocks(theme="soft", title="CrediTrust Complaint Assistant") as demo:
    with gr.Row():
        gr.HTML("<h1 style='text-align:center; color:#2c3e50;'>🏦 CrediTrust Complaint Assistant</h1>")
    with gr.Row():
        gr.Markdown(
            "Welcome to **CrediTrust Complaint Analyzer**.\n\n"
            "💬 Ask me questions about customer complaints, and I’ll retrieve the most relevant cases, "
            "then summarize insights for you. \n\n"
            "Our goal: **transparency & trust in financial services.**"
        )

    chatbot = gr.Chatbot(
        height=400,
        bubble_full_width=False,
        show_label=False,
    )

    with gr.Row():
        query_input = gr.Textbox(
            placeholder="E.g. What issues are customers facing with mortgages?",
            show_label=False,
            scale=8,
        )
        ask_button = gr.Button("🔍 Ask", scale=1)
        clear_button = gr.Button("🧹 Clear", scale=1)

    # Button actions
    ask_button.click(rag_chatbot_interface, [query_input, chatbot], chatbot)
    clear_button.click(lambda: [], None, chatbot)

# ----------------------------
# Run App
# ----------------------------

if __name__ == "__main__":
    demo.launch()
