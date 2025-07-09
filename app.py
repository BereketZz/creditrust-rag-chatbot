import gradio as gr
from src.rag_pipeline import generate_answer

# ----------------------------
# RAG Chat Interface Functions
# ----------------------------

def rag_chatbot_interface(query):
    if not query.strip():
        return "Please enter a valid question.", []

    try:
        answer, sources = generate_answer(query)
    except Exception as e:
        return f"Error: {str(e)}", []

    # Limit to first 3 sources
    source_display = [f"📄 Source {i+1}: {chunk[:300]}..." for i, chunk in enumerate(sources[:3])]
    return answer, source_display

# ----------------------------
# Gradio Interface Layout
# ----------------------------

with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("## 🏦 CrediTrust Complaint Analyzer")
    gr.Markdown("Ask a question about customer complaints. The system will use retrieved complaint texts to answer.")

    with gr.Row():
        query_input = gr.Textbox(label="Your Question", placeholder="E.g. What issues are customers facing with mortgages?")
    
    ask_button = gr.Button("🔍 Ask")
    clear_button = gr.Button("🧹 Clear")

    answer_output = gr.Textbox(label="AI-Generated Answer", lines=4)
    source_output = gr.HighlightedText(label="📚 Top Source Chunks")

    # Button Actions
    ask_button.click(fn=rag_chatbot_interface, inputs=[query_input], outputs=[answer_output, source_output])
    clear_button.click(fn=lambda: ("", []), inputs=[], outputs=[answer_output, source_output, query_input])

# ----------------------------
# Run App
# ----------------------------

if __name__ == "__main__":
    demo.launch()
