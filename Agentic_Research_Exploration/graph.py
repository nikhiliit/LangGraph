from typing import Annotated, List, Any, Dict
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
import sqlite3


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    pdf_content: str
    question: str
    user_registered: bool


class ResearchAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        self.tools = create_research_tools()
        self.model_name = model_name
        self.llm_with_tools = None
        self.graph = None
        self.memory = None

    def setup(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

        model_config = LLM_MODELS.get(self.model_name, LLM_MODELS["gpt-4o-mini"])
        llm = ChatOpenAI(model=model_config["model"])
        self.llm_with_tools = llm.bind_tools(self.tools)

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

        # Check if content is small enough to process entirely
        if len(pdf_content) <= MAX_CHUNK_SIZE:
            # Process entire document
            system_prompt = f"""You are a research assistant analyzing a scientific paper.

                Full Paper Content:
                {pdf_content}

                Question: {question}

                Please provide a comprehensive answer based on the entire paper content.
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

                Question: {question}

                Based on this overview, provide a comprehensive answer. If you need more specific sections, mention which parts would be most relevant.
                """
            messages = [HumanMessage(content=summary_prompt)]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

    def router(self, state: State) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "END"

    def tool_handler(self, state: State) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        if last_message.content and "pdf_content" not in last_message.content.lower():
            return {"pdf_content": last_message.content}
        return {}

    def build_graph(self):
        graph_builder = StateGraph(State)

        graph_builder.add_node("researcher", self.research_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))

        graph_builder.add_conditional_edges("researcher", self.router, {"tools": "tools", "END": END})
        graph_builder.add_edge("tools", "researcher")
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

        initial_state = {
            "messages": [HumanMessage(content=f"Analyze PDF: {pdf_path}")],
            "pdf_content": pdf_content,
            "question": question,
            "user_registered": user_registered
        }

        # Use config with thread_id only if checkpointing is available
        if self.memory:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(initial_state, config=config)
        else:
            result = self.graph.invoke(initial_state)
        return result["messages"][-1].content
