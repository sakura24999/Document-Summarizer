from .base import BaseDocumentProcessor
from .docx_processor import DOCXProcessor
from .pdf_processor import PDFProcessor
from .text_processor import TEXTProcessor

__all__ = ['BaseDocumentProcessor', 'DOCXProcessor', 'PDFProcessor', 'TEXTProcessor']

def get_processor_for_file(file_path):
    """
    ファイルパスに基づいて適切なプロセッサーを返します

    Args:
        file_path (str): 処理するファイルのパス

    Returns:
        BaseDocumentProcessor: 適切なプロセッサーのインスタンス
    """
    if file_path.endswith('.pdf'):
        return PDFProcessor()
    elif file_path.endswith('.docx'):
        return DOCXProcessor()
    elif file_path.endswith('.txt'):
        return TEXTProcessor()
    else:
        raise ValueError(f"サポートされていないファイル形式: {file_path}")
