import asyncio
import sys
from graph import ResearchAgent
from config import OPENAI_API_KEY


def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    if len(sys.argv) < 3:
        print("Usage: python main.py <pdf_path> <question>")
        print("Example: python main.py paper.pdf 'What is the main contribution?'")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    agent = ResearchAgent()
    agent.setup()

    print(f"Processing PDF: {pdf_path}")
    print(f"Question: {question}")
    print("-" * 50)

    result = agent.process_question(pdf_path, question)
    print(result)


if __name__ == "__main__":
    main()
