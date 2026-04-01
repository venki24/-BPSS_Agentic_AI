from app.agent.tools.pdf_tool import search_pdfs
from app.agent.tools.csv_tool import query_csv
from app.agent.tools.docx_tool import extract_docx

TOOLS = [search_pdfs, query_csv, extract_docx]

__all__ = ["search_pdfs", "query_csv", "extract_docx", "TOOLS"]
