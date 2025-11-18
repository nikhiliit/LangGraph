#!/usr/bin/env python3
"""
Quick launcher for the Research Paper Analysis Gradio interface.
"""

from app import create_gradio_interface

if __name__ == "__main__":
    print("🚀 Starting Research Paper Analysis Agent...")
    print("📄 Upload a PDF and ask questions about research papers!")
    print("🌐 Interface will be available at: http://localhost:7860")
    print("❌ Press Ctrl+C to stop the server")
    print("-" * 50)

    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False  # Set to True if you want a public link
    )
