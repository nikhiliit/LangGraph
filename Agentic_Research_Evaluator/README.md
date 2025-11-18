---
title: Agentic_Research_Evaluator
app_file: app.py
sdk: gradio
sdk_version: 5.34.2
---
# Enhanced Research Paper Analysis Agent with Evaluator Node

A hallucination-preventing research assistant built with LangGraph, following the **Sidekick pattern** from lab4.ipynb. This enhanced version adds an evaluator node that validates every response against the source PDF content.

## 🧠 What's New: Hallucination Prevention

Unlike the original single-node agent, this version includes:
- **Evaluator Node**: Validates responses for accuracy and grounding
- **Iterative Refinement**: Failed evaluations trigger response improvement
- **PDF Content Validation**: Ensures answers are based solely on provided documents
- **Structured Feedback**: Pydantic models provide consistent evaluation criteria

## 🏗️ Architecture: Sidekick Pattern

Following the exact same pattern from LangGraph lab4.ipynb:

```
START → Researcher → (Tools OR Evaluator) → (Continue OR END)
                     ↓
                Tool Results
                     ↓
               Back to Researcher
                     ↓
                 Evaluator
                 ↙        ↘
          Continue? → YES    NO → Back to Researcher
               ↓             ↓
              END     Iterative Improvement
```

### Key Components

#### **Researcher Node**
- Analyzes PDF content with tool integration
- Receives feedback from evaluator for improvement
- Processes both small and large documents with chunking

#### **Evaluator Node**
- **Pydantic Structured Output**: Consistent evaluation schema
- **PDF Content Validation**: Checks grounding against source material
- **Hallucination Detection**: Identifies unverified information
- **Feedback Generation**: Provides specific improvement suggestions

#### **State Management**
```python
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    pdf_content: str
    question: str
    user_registered: bool
    # NEW: Evaluator fields
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd /Users/nikk/Desktop/projects/Agentic_Research_Evaluator
python run_evaluator_agent.py
```

The launcher automatically:
- Creates a `.venv` virtual environment
- Installs all dependencies
- Starts the web interface

### 2. Set Environment Variables
```bash
cp env_example .env
# Edit .env with your OpenAI API key
```

### 3. Run the Agent
```bash
# Web interface (recommended)
python run_evaluator_agent.py

# CLI mode
python main.py path/to/paper.pdf "What is the main contribution?"
```

## 🔍 How the Evaluator Works

### Evaluation Criteria
1. **Accuracy**: Information matches PDF content exactly
2. **Grounding**: No external knowledge or assumptions
3. **Relevance**: Directly answers the question asked
4. **Completeness**: Addresses question based on available content
5. **Limitations**: Acknowledges when paper lacks information

### Super-Step Process
```
User Question → Research Analysis → Evaluator Validation
                                           ↓
                                    Criteria Met?
                                 ↙           ↘
                            YES → END    NO → Feedback to Researcher
                                 ↓              ↓
                            Final Answer    Improved Analysis → Evaluator...
```

## 📋 Example Scenarios

### ✅ Accurate Response (Evaluator Approves)
```
Question: "What datasets were used?"
PDF Content: "We evaluated on ImageNet, CIFAR-10, and MNIST datasets..."
Response: "The paper evaluates on ImageNet, CIFAR-10, and MNIST datasets."
→ ✅ Success criteria met, response grounded in PDF
```

### ❌ Hallucination Detected (Evaluator Rejects)
```
Question: "How does this compare to BERT?"
PDF Content: "Our method achieves 85% accuracy..." (no BERT mention)
Response: "This approach outperforms BERT by 10%..."
→ ❌ Hallucination detected, sent back for correction
```

## 🛠️ Technical Details

### Dependencies Added
```
pydantic>=2.0.0  # For structured evaluator output
```

### Router Functions
- `research_router()`: Routes after research (tools vs evaluator)
- `evaluation_router()`: Routes after evaluation (continue vs end)

### Checkpointing
- SQLite-based conversation persistence
- Maintains evaluation history across sessions

## 🧪 Testing & Debugging

### Debug Evaluation Execution
Inspect the evaluator node step-by-step to see exactly how it works:
```bash
python debug_evaluator.py <pdf_path> <question>
# Example
python debug_evaluator.py research_paper.pdf "What is the main contribution?"
```

This shows:
- Research node response generation
- Router decision (tools vs evaluator)
- Detailed evaluation feedback
- Success criteria assessment
- Final routing decision

### Test Hallucination Detection
Run without arguments to see test cases for different prompt types:
```bash
python debug_evaluator.py
```

### Web Interface Testing
Use the full Gradio interface to see evaluation in action:
```bash
python run_evaluator_agent.py
```
Look for 🔍 evaluation messages in the chat after asking questions.

## 🔧 Development

### Virtual Environment
The launcher script automatically manages the `.venv` environment:
- Creates virtual environment if needed
- Installs/updates dependencies
- Activates environment for debugging

### Debugging Tips
- Check evaluation feedback in logs
- Monitor `success_criteria_met` flag
- Review `feedback_on_work` for improvement suggestions

## 📊 Benefits Over Original Agent

| Feature | Original Agent | Enhanced with Evaluator |
|---------|---------------|------------------------|
| Hallucination Prevention | ❌ None | ✅ Built-in validation |
| Response Accuracy | ⚠️ Variable | ✅ Guaranteed grounding |
| Iterative Improvement | ❌ Single shot | ✅ Feedback loop |
| PDF Content Validation | ❌ None | ✅ Every response checked |
| Structured Evaluation | ❌ None | ✅ Pydantic schemas |

## 🎯 Use Cases

- **Research Analysis**: Ensure accurate paper summaries
- **Literature Review**: Prevent cross-contamination between papers
- **Academic Q&A**: Maintain source fidelity
- **Document Analysis**: Validate against specific content

---

**Built following LangGraph Sidekick pattern for maximum reliability and accuracy.**
