from typing import Annotated, List, Any, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
# Try to import SqliteSaver, fallback to MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    print("Using SqliteSaver for checkpointing")
except ImportError:
    try:
        from langgraph.checkpoint.memory import MemorySaver as SqliteSaver
        print("SqliteSaver not available, using MemorySaver as fallback")
    except ImportError:
        print("Warning: No checkpointing available, using in-memory storage only")
        SqliteSaver = None
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from tools import create_research_tools, chunk_text
from config import DB_PATH, MAX_CHUNK_SIZE, LLM_MODELS, OPENAI_API_KEY
from pydantic import BaseModel, Field
import sqlite3


# NEW: Structured output for evaluator following lab4 pattern
class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the research assistant's response accuracy and grounding")
    success_criteria_met: bool = Field(description="Whether the response is accurate, based solely on PDF content, and answers the question")
    user_input_needed: bool = Field(description="True if clarification needed, agent is hallucinating, or cannot answer from PDF content")


# ENHANCED: State with evaluator fields following lab4 sidekick pattern
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


class ResearchAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        self.tools = create_research_tools()
        self.model_name = model_name
        self.llm_with_tools = None
        # NEW: Separate evaluator LLM following lab4 pattern
        self.evaluator_llm = None
        self.graph = None
        self.memory = None

    def setup(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

        model_config = LLM_MODELS.get(self.model_name, LLM_MODELS["gpt-4o-mini"])
        llm = ChatOpenAI(model=model_config["model"])
        self.llm_with_tools = llm.bind_tools(self.tools)

        # NEW: Initialize evaluator LLM with structured output
        evaluator_model = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm = evaluator_model.with_structured_output(EvaluatorOutput)

        # Set up checkpointing
        if SqliteSaver:
            try:
                # Try to use SqliteSaver with connection (for actual sqlite checkpointing)
                conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                self.memory = SqliteSaver(conn)
                print("Using SqliteSaver for persistent checkpointing")
            except TypeError:
                # MemorySaver doesn't take connection argument
                self.memory = SqliteSaver()
                print("Using MemorySaver for in-memory checkpointing")
        else:
            self.memory = None
            print("Warning: No checkpointing available")

        self.build_graph()

    def research_node(self, state: State) -> Dict[str, Any]:
        # Check user registration first
        if not state.get("user_registered", False):
            return {"messages": [{"role": "assistant", "content": "⚠️ Please register first using the registration form in the sidebar before accessing the chatbot."}]}

        if not state.get("pdf_content"):
            return {"messages": [{"role": "assistant", "content": "Please extract a PDF first using the extract_pdf_text tool."}]}

        from config import MAX_CHUNK_SIZE  # Import here to ensure availability

        pdf_content = state["pdf_content"]
        question = state.get('question', 'Summarize this paper')

        # NEW: Add success criteria context
        success_criteria = state.get("success_criteria",
            "Response must be accurate, based solely on PDF content, directly answer the question, and contain no hallucinations")

        # NEW: Include previous feedback if available (following lab4 pattern)
        feedback_context = ""
        if state.get("feedback_on_work"):
            feedback_context = f"""

        PREVIOUS EVALUATION FEEDBACK: {state['feedback_on_work']}
        Please address this feedback and ensure your response meets the success criteria."""

        # Check if content is small enough to process entirely
        if len(pdf_content) <= MAX_CHUNK_SIZE:
            # Process entire document
            system_prompt = f"""You are a research assistant analyzing a scientific paper.

            SUCCESS CRITERIA: {success_criteria}{feedback_context}

            Full Paper Content:
            {pdf_content}

            Question: {question}

            IMPORTANT: Your response must be based SOLELY on the provided PDF content above.
            Do not include any information, assumptions, or knowledge not present in this specific paper.
            If the paper doesn't contain information to answer the question, clearly state this limitation.
            """
            messages = [HumanMessage(content=system_prompt)]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        else:
            # For large documents, use intelligent retrieval
            chunks = chunk_text(pdf_content)

            # Create a summary of the paper first for context
            summary_prompt = f"""You are analyzing a research paper. Here are the first few chunks:

            {chunks[0]}
            {chunks[1] if len(chunks) > 1 else ""}

            SUCCESS CRITERIA: {success_criteria}{feedback_context}

            Question: {question}

            Based on this overview, provide a comprehensive answer. If you need more specific sections, mention which parts would be most relevant.
            IMPORTANT: Base your response only on the provided paper content - no external knowledge or assumptions.
            """
            messages = [HumanMessage(content=summary_prompt)]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

    # NEW: Evaluator node following lab4 sidekick pattern
    def evaluator_node(self, state: State) -> Dict[str, Any]:
        """Evaluator node that validates research responses against PDF content."""
        last_response = state["messages"][-1].content
        pdf_content = state["pdf_content"]
        question = state.get("question", "")
        success_criteria = state.get("success_criteria", "")

        system_message = """You are an evaluator that determines if a research analysis response is accurate and properly grounded in the PDF content.

        Evaluate for:
        1. ACCURACY: Information matches what's actually in the PDF
        2. GROUNDING: No hallucinated information not present in the source
        3. RELEVANCE: Response directly answers the question asked
        4. COMPLETENESS: Addresses the question adequately based on available content
        5. LIMITATIONS: Clearly states if paper lacks information to fully answer

        Be strict: if the response contains any information not explicitly stated in the PDF, mark as not meeting criteria."""

        # Create a representative sample of the PDF content for evaluation
        pdf_sample = pdf_content[:3000] + "..." if len(pdf_content) > 3000 else pdf_content

        user_message = f"""Evaluate this research analysis response:

        QUESTION ASKED: {question}

        SUCCESS CRITERIA: {success_criteria}

        PDF CONTENT SAMPLE: {pdf_sample}

        ASSISTANT RESPONSE TO EVALUATE: {last_response}

        Analyze whether this response meets the success criteria. Check specifically:
        - Is all information present in the PDF content sample?
        - Does it directly answer the question?
        - Are there any hallucinations or external assumptions?
        - Does it acknowledge limitations if the paper lacks relevant information?"""

        if state.get("feedback_on_work"):
            user_message += f"\n\nPREVIOUS FEEDBACK: {state['feedback_on_work']}\nAddress whether this feedback has been adequately resolved."

        from langchain_core.messages import SystemMessage
        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]

        eval_result = self.evaluator_llm.invoke(evaluator_messages)

        # Return evaluation results following lab4 pattern
        new_state = {
            "messages": [{"role": "assistant", "content": f"🔍 Evaluation: {eval_result.feedback}"}],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed
        }
        return new_state

    def research_router(self, state: State) -> str:
        """Route after research node - tools or evaluator (following lab4 pattern)."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "evaluator"

    def evaluation_router(self, state: State) -> str:
        """Route after evaluator - continue research or end (following lab4 pattern)."""
        if state["success_criteria_met"] or state["user_input_needed"]:
            return END
        return "researcher"  # Back to research node for correction

    def tool_handler(self, state: State) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        if last_message.content and "pdf_content" not in last_message.content.lower():
            return {"pdf_content": last_message.content}
        return {}

    def build_graph(self):
        graph_builder = StateGraph(State)

        # Add nodes: researcher, tools, evaluator
        graph_builder.add_node("researcher", self.research_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator_node)  # NEW

        # Add conditional edges following lab4 sidekick pattern
        graph_builder.add_conditional_edges(
            "researcher", self.research_router,
            {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "researcher")
        graph_builder.add_conditional_edges(
            "evaluator", self.evaluation_router,
            {"researcher": "researcher", END: END}
        )
        graph_builder.add_edge(START, "researcher")

        # Compile with checkpointing if available
        if self.memory:
            self.graph = graph_builder.compile(checkpointer=self.memory)
        else:
            self.graph = graph_builder.compile()

    def process_question(self, pdf_path: str, question: str, thread_id: str = "default", user_registered: bool = False) -> str:
        pdf_content = ""
        for tool in self.tools:
            if tool.name == "extract_pdf_text":
                pdf_content = tool.func(pdf_path)
                break

        if "Error:" in pdf_content:
            return pdf_content

        # ENHANCED: Initialize state with evaluator fields
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze PDF: {pdf_path}")],
            "pdf_content": pdf_content,
            "question": question,
            "user_registered": user_registered,
            "success_criteria": "Response must be accurate, based solely on PDF content, directly answer the question, and contain no hallucinations",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False
        }

        # Use config with thread_id only if checkpointing is available
        if self.memory:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(initial_state, config=config)
        else:
            result = self.graph.invoke(initial_state)

        # Return the final response (before evaluation feedback)
        final_messages = result["messages"]
        # Find the last substantive response (not evaluation feedback)
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("🔍 Evaluation:"):
                return msg.content

        return result["messages"][-1].content
