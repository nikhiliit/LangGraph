---
title: Agentic_Research_Exploration
app_file: app.py
sdk: gradio
sdk_version: 5.34.2
---
# Research Paper Analysis Agent

A LangGraph-based agent for extracting and analyzing research papers using natural language questions. Built following the sidekick architecture pattern with SQLite checkpointing and tool-based interactions.

## Features

- 🌐 **Professional Web Interface**: Clean ChatInterface design with sidebar controls
- 📄 PDF text extraction and processing with metadata analysis
- 🧠 Chunk-based content analysis for large research documents
- 💾 SQLite-based conversation memory and checkpointing persistence
- 🔧 Tool-based research assistance (PDF extraction, file management)
- 🎯 Single-node LangGraph architecture following sidekick patterns
- ❓ Natural language Q&A about research papers with contextual responses
- 💡 **Question Templates**: Pre-built question suggestions for comprehensive paper analysis
- 🔢 **LaTeX/MathJax Support**: Renders mathematical equations beautifully in the web interface

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**

   **For Local Development:**
   ```bash
   cp env_example .env
   # Edit .env and add your API keys:
   # - OPENAI_API_KEY (required)
   # - PUSHOVER_TOKEN and PUSHOVER_USER (optional, for notifications)
   ```

   **For Hugging Face Spaces Deployment:**
   - Go to your Space settings → Secrets
   - Add these secrets:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `PUSHOVER_TOKEN`: Your Pushover token (optional)
     - `PUSHOVER_USER`: Your Pushover user key (optional)
   - The app automatically detects HF Spaces and uses secrets instead of .env file

3. **Configure Pushover (Optional):**
   - Visit [Pushover.net](https://pushover.net/) and create an account
   - Create an application to get your API token
   - Add `PUSHOVER_TOKEN` and `PUSHOVER_USER` to your `.env` file (local) or HF Space secrets (deployment)
   - This enables notifications when new users register

4. **Verify installation:**
```bash
python test_agent.py
```

## Hugging Face Spaces Deployment

1. **Create a new Space:**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Create a new Space with Gradio SDK
   - Choose "Gradio" as the SDK

2. **Upload your code:**
   - Clone this repository to your Space
   - Or upload the files directly

3. **Configure Secrets:**
   - Go to your Space → Settings → Secrets
   - Add required secrets (see Setup section above)

4. **Set the App File:**
   - In Space settings, set the App file to: `app.py`
   - The app will automatically start when you push the code

5. **Access your Space:**
   - Your research paper analysis agent will be available at: `https://[your-username]-[space-name].hf.space`

## Usage

### User Registration
**Important**: Before using the chatbot, users must register by providing their name and email address. This is required because:

- OpenAI models are paid services
- User contact information is needed to keep track of your usage
- Push notifications can be sent for important updates

### Web Interface (Recommended)
The easiest way to use the agent is through the interactive web interface with a clean, professional design:

```bash
python run_gradio.py
```
Then open your browser to `http://localhost:7860` to enjoy:

#### **🎨 Clean UI Design**
- **Main Chat Area**: Focused conversation interface with LaTeX equation rendering
- **Sidebar Controls**: Organized user registration, PDF upload and settings without clutter
- **Professional Layout**: Inspired by modern chatbot interfaces

#### **📋 Key Features**
- 👤 **User Registration**: Required registration form to keep track of usage
- 📤 **PDF Upload**: Drag-and-drop PDF files in the sidebar
- 💬 **Natural Chat**: Conversational Q&A about research papers (after registration)
- 🔢 **LaTeX Rendering**: Beautiful mathematical equations in responses
- 💡 **Question Templates**: Collapsible accordion with 9+ question suggestions
- 🚀 **Quick Examples**: Clickable examples for instant questions
- 🔄 **Context Memory**: Maintains conversation across multiple exchanges
- 🔄 **Easy Reset**: Clear conversations when switching papers
- 📱 **Push Notifications**: Optional push notifications for new registrations

### Command Line Interface
For programmatic use or automation:

```bash
python main.py path/to/paper.pdf "What is the main contribution?"
```

### Interactive Demo
Enhanced command-line experience with formatted output:

```bash
python demo.py path/to/paper.pdf "Summarize the methodology"
```

### Example Questions
- "What is the main contribution of this paper?"
- "Summarize the experimental results"
- "How does this approach compare to previous work?"
- "What datasets were used?"
- "What are the limitations mentioned?"

## Architecture

### **config.py**
Environment variables and configuration settings:
- OpenAI API key
- Database path
- Chunk size and overlap parameters

### **tools.py**
PDF processing and utility tools:
- `extract_pdf_text()`: Extract text from PDF files
- `get_pdf_info()`: Get PDF metadata
- `chunk_text()`: Split text into manageable chunks
- `create_research_tools()`: LangChain tool wrappers

### **graph.py**
LangGraph implementation with research agent:
- `ResearchAgent` class with single-node architecture
- SQLite checkpointing for conversation persistence
- Tool integration and routing logic
- State management for PDF content and Q&A workflow

### **main.py**
Command-line interface for basic usage

### **demo.py**
Enhanced demo script with formatted output

### **app.py**
Interactive Gradio web interface with PDF upload and conversational chat

### **run_gradio.py**
Simple launcher script for the Gradio web interface

## Project Structure

```
Agentic_Research_Exploration/
├── config.py              # Configuration settings
├── tools.py               # PDF processing tools
├── graph.py               # LangGraph agent implementation
├── main.py                # CLI interface
├── demo.py                # Demo script
├── app.py                 # Gradio web interface
├── run_gradio.py          # Gradio launcher script
├── test_gradio.py         # Gradio interface tests
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (copied from parent)
└── research_agent.db     # SQLite database (created automatically)
```

## Checkpointing

The application supports conversation persistence through LangGraph's checkpointing system:

- **SqliteSaver**: Used when available for persistent SQLite-based storage
- **MemorySaver**: Fallback for in-memory storage when SqliteSaver is unavailable
- **No Checkpointing**: Graceful degradation if neither is available

This ensures compatibility across different deployment environments (local, HF Spaces, etc.).

## Testing

Run the test suite:
```bash
python test_agent.py          # Basic functionality tests
python test_with_mock.py      # Full pipeline with mock data
```

## Code Quality

- ✅ PEP-8 compliant
- ✅ Minimal comments following LangGraph patterns
- ✅ Type hints and proper imports
- ✅ Modular, interdependent Python scripts
- ✅ No linting errors

## Architecture Notes

- **Single Node Design**: Unlike multi-agent systems, uses one node with tools
- **SQLite Checkpointing**: Persistent memory across sessions
- **Chunk-based Processing**: Handles large PDFs by breaking into chunks
- **Tool-first Approach**: Research tasks handled through specialized tools
- **State Management**: TypedDict-based state following LangGraph patterns
