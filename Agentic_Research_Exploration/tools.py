import PyPDF2
import os
import requests
from typing import List, Dict, Any
from langchain_core.tools import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from config import MAX_CHUNK_SIZE, OVERLAP_SIZE, PUSHOVER_TOKEN, PUSHOVER_USER


def extract_pdf_text(file_path: str) -> str:
    """Extract text content from a PDF file."""
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found"

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> List[str]:
    """Split text into overlapping chunks for processing."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            end = text.rfind(' ', start, end) or end
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def get_pdf_info(file_path: str) -> str:
    """Get basic information about a PDF file."""
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found"

    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            info = pdf_reader.metadata
            return f"""PDF Information:
Title: {info.title or 'Unknown'}
Author: {info.author or 'Unknown'}
Pages: {len(pdf_reader.pages)}
Subject: {info.subject or 'Unknown'}"""
    except Exception as e:
        return f"Error getting PDF info: {str(e)}"


def send_push_notification(text: str) -> str:
    """Send a push notification to the user via Pushover"""
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        return "Pushover not configured - PUSHOVER_TOKEN and PUSHOVER_USER environment variables required"

    try:
        pushover_url = "https://api.pushover.net/1/messages.json"
        response = requests.post(pushover_url, data={
            "token": PUSHOVER_TOKEN,
            "user": PUSHOVER_USER,
            "message": text
        })
        if response.status_code == 200:
            return "Push notification sent successfully"
        else:
            return f"Failed to send push notification: {response.text}"
    except Exception as e:
        return f"Error sending push notification: {str(e)}"


def create_research_tools() -> List[Tool]:
    """Create tools for research paper processing."""
    pdf_extract_tool = Tool(
        name="extract_pdf_text",
        func=extract_pdf_text,
        description="Extract text content from a PDF research paper. Input: file path to PDF"
    )

    pdf_info_tool = Tool(
        name="get_pdf_info",
        func=get_pdf_info,
        description="Get metadata information about a PDF file. Input: file path to PDF"
    )

    push_notification_tool = Tool(
        name="send_push_notification",
        func=send_push_notification,
        description="Send a push notification to the user. Input: message text to send"
    )

    file_tools = FileManagementToolkit(root_dir=".").get_tools()

    return [pdf_extract_tool, pdf_info_tool, push_notification_tool] + file_tools
