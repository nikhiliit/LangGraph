import gradio as gr
import tempfile
import os
from graph import ResearchAgent
from config import OPENAI_API_KEY, LLM_MODELS
from user_manager import UserManager
from tools import send_push_notification
import uuid


class ResearchChatInterface:
    def __init__(self, model_name="gpt-4o-mini"):
        self.agent = None
        self.model_name = model_name
        self.uploaded_pdf_path = None
        self.user_manager = UserManager()
        self.registered_user = None

    def initialize_agent(self):
        """Initialize the research agent if not already done."""
        if self.agent is None:
            self.agent = ResearchAgent(self.model_name)
            self.agent.setup()

    def switch_model(self, new_model_name):
        """Switch to a different OpenAI model."""
        if new_model_name != self.model_name:
            self.model_name = new_model_name
            self.agent = None  # Force re-initialization
            self.initialize_agent()
            return f"✅ Switched to {new_model_name}"
        return f"Already using {new_model_name}"

    def upload_pdf(self, file):
        """Handle PDF upload and extract content."""
        if file is None:
            return "Please upload a PDF file first."

        try:
            self.initialize_agent()

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                with open(file.name, 'rb') as src_file:
                    tmp_file.write(src_file.read())
                self.uploaded_pdf_path = tmp_file.name

            # Generate unique session ID for this PDF
            self.session_id = str(uuid.uuid4())

            # Extract basic info about the PDF
            pdf_info = ""
            for tool in self.agent.tools:
                if tool.name == "get_pdf_info":
                    pdf_info = tool.func(self.uploaded_pdf_path)
                    break

            return f"✅ PDF uploaded successfully!\n\n{pdf_info}\n\nYou can now ask questions about this research paper."

        except Exception as e:
            return f"❌ Error uploading PDF: {str(e)}"

    def chat_response(self, message, history):
        """Process chat messages and respond using the research agent."""
        # Add user message to history
        user_msg = {"role": "user", "content": message}
        current_history = history + [user_msg]

        if not self.uploaded_pdf_path:
            response_msg = {"role": "assistant", "content": "Please upload a PDF file first before asking questions."}
            return current_history + [response_msg]

        if not message.strip():
            response_msg = {"role": "assistant", "content": "Please ask a question about the research paper."}
            return current_history + [response_msg]

        try:
            self.initialize_agent()

            # Use the existing process_question method
            user_registered = self.check_user_registration()
            response = self.agent.process_question(
                self.uploaded_pdf_path,
                message,
                thread_id=self.session_id,
                user_registered=user_registered
            )

            response_msg = {"role": "assistant", "content": response}
            return current_history + [response_msg]

        except Exception as e:
            error_msg = {"role": "assistant", "content": f"❌ Error processing question: {str(e)}"}
            return current_history + [error_msg]

    def register_user(self, name: str, email: str) -> str:
        """Register a user with name and email."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

        success, message = self.user_manager.register_user(self.session_id, name, email)

        if success:
            self.registered_user = self.user_manager.get_user(self.session_id)
            # Send push notification about new registration
            try:
                send_push_notification(f"New user registered: {name} ({email})")
            except:
                pass  # Don't fail registration if push fails

        return message

    def check_user_registration(self) -> bool:
        """Check if current session has a registered user."""
        if not self.session_id:
            return False
        self.registered_user = self.user_manager.get_user(self.session_id)
        return self.registered_user is not None

    def reset_conversation(self):
        """Reset the conversation and uploaded PDF."""
        if self.uploaded_pdf_path and os.path.exists(self.uploaded_pdf_path):
            try:
                os.unlink(self.uploaded_pdf_path)
            except:
                pass  # Ignore cleanup errors

        self.uploaded_pdf_path = None
        self.session_id = None
        self.registered_user = None
        return None, [], "Conversation reset. Upload a new PDF to start."


def create_gradio_interface():
    """Create and configure the Gradio interface using ChatInterface with sidebar."""
    interface = ResearchChatInterface()

    # Session state management
    session_state = gr.State()

    def chat_with_pdf(message, history, model_selection, template_question, session_id):
        """Main chat function that handles PDF analysis."""
        # Set the session_id on the interface
        interface.session_id = session_id

        # Switch model if needed
        if model_selection != interface.model_name:
            try:
                switch_msg = interface.switch_model(model_selection)
                # Return a confirmation message about model switch
                return f"{switch_msg}\n\nNow analyzing with {model_selection}. You can ask your question again."
            except ValueError as e:
                return f"❌ Model switch failed: {str(e)}\n\nPlease select a different model or ensure you have the required API keys set up."

        # If a template question is selected, use it instead of the typed message
        if template_question:
            message = template_question

        response_list = interface.chat_response(message, history)
        # Extract the last assistant message content for ChatInterface
        if response_list and len(response_list) > 0:
            last_message = response_list[-1]
            if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                return last_message["content"]
        return "I apologize, but I encountered an error processing your question."

    def register_user_form(name, email, session_id):
        """Handle user registration."""
        interface.session_id = session_id
        result = interface.register_user(name, email)
        return result, interface.session_id

    def upload_and_status(file, session_id):
        """Handle PDF upload and return status."""
        interface.session_id = session_id
        status = interface.upload_pdf(file)
        return status, session_id

    def initialize_session(session_id):
        """Initialize session if not exists."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        return session_id

    def reset_all(session_id):
        """Reset conversation and return empty states."""
        interface.session_id = session_id
        result = interface.reset_conversation()
        return result[0], [], result[2], "", session_id  # pdf, chatbot, status, registration_status, session_id

    # Create the main ChatInterface
    chatbot = gr.ChatInterface(
        chat_with_pdf,
        type="messages",
        title="🔬 Research Paper Analysis Agent",
        description="Upload a PDF and ask questions about research papers. Supports LaTeX equations. Powered by LangGraph, LangChain & LangSmith.",
        theme=gr.themes.Soft(primary_hue="blue"),
        textbox=gr.Textbox(
            placeholder="Ask about the research paper...",
            show_label=False
        ),
        additional_inputs=[
            gr.Dropdown(
                label="🤖 LLM Model",
                choices=list(LLM_MODELS.keys()),
                value="gpt-4o-mini",
                info="Select the AI model to use for analysis"
            ),
            gr.Dropdown(
                label="💡 Question Templates",
                choices=[
                    "What is the main contribution of the paper?",
                    "What are the core mathematical foundations of the paper?",
                    "What is the main mathematical proposal?",
                    "What is the reasoning for the proposal?",
                    "Summarize the methodology and approach",
                    "What datasets or experiments were used?",
                    "How does this compare to previous work?",
                    "What are the key results and findings?",
                    "What are the limitations and future work?"
                ],
                value=None,
                info="Select a question template to get started"
            ),
            session_state
        ],
        additional_inputs_accordion="Settings & Templates",
        examples=[
            ["What is the main contribution of this paper?"],
            ["What are the core mathematical foundations?"],
            ["Summarize the methodology"]
        ]
    )

    # Configure LaTeX rendering for the chatbot
    chatbot.chatbot.latex_delimiters = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
        {"left": "\\(", "right": "\\)", "display": False}
    ]

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        # Initialize session on load
        demo.load(
            fn=initialize_session,
            inputs=[session_state],
            outputs=[session_state]
        )

        with gr.Sidebar():
            gr.Markdown("## 👤 User Registration")
            gr.Markdown("**Required**: Please register before using the chatbot. Since OpenAI models are paid services, we need your contact information to keep track of your usage.")

            name_input = gr.Textbox(
                label="Full Name",
                placeholder="Enter your full name",
                lines=1
            )

            email_input = gr.Textbox(
                label="Email Address",
                placeholder="Enter your email address",
                lines=1
            )

            register_btn = gr.Button("Register", variant="primary")
            registration_status = gr.Textbox(
                label="Registration Status",
                interactive=False,
                lines=2,
                placeholder="Please register to continue..."
            )

            gr.Markdown("---")
            gr.Markdown("## 📄 Document Upload")

            pdf_upload = gr.File(
                label="Upload Research Paper PDF",
                file_types=[".pdf"],
                height=80
            )

            upload_status = gr.Textbox(
                label="📋 Upload Status",
                interactive=False,
                lines=4,
                placeholder="Upload a PDF to get started..."
            )

            gr.Markdown("---")

            reset_btn = gr.Button("🔄 Reset Conversation", variant="stop")

            gr.Markdown("### 🔧 Tech Stack")
            gr.Markdown("""
**Built with:**
- 🤖 **LangGraph**: Agent orchestration & state management
- 🦜 **LangChain**: LLM integration & tool binding
- 📊 **LangSmith**: Monitoring & observability
- 🎨 **Gradio**: Web interface & UI
- 💾 **SQLite**: Persistent conversation memory

**🤖 AI Models:**
- ✅ **GPT-4o-mini**: Ready to use (requires OpenAI API key)
""")

            gr.Markdown("### ℹ️ How to Use")
            gr.Markdown("""
            1. **Upload** a PDF research paper
            2. **Select** your preferred AI model
            3. **Ask questions** using templates or type your own
            4. **View responses** with rendered LaTeX equations
            5. **Continue** the conversation for deeper analysis
            """)

        # Render the main chatbot interface
        chatbot.render()

        # Event handlers
        register_btn.click(
            fn=register_user_form,
            inputs=[name_input, email_input, session_state],
            outputs=[registration_status, session_state]
        )

        pdf_upload.upload(
            fn=upload_and_status,
            inputs=[pdf_upload, session_state],
            outputs=[upload_status, session_state]
        )

        reset_btn.click(
            fn=reset_all,
            inputs=[session_state],
            outputs=[pdf_upload, chatbot.chatbot, upload_status, registration_status, session_state]
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
