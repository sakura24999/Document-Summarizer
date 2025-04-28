import re
from typing import Dict, List, Any, Optional
import PyPDF2
from datetime import datetime
from .base import BaseDocumentProcessor


class PDFProcessor(BaseDocumentProcessor):
    """
    PDFファイルを処理するためのプロセッサークラス
    """

    def __init__(self):
        """
        PDFプロセッサーの初期化
        """
        super().__init__()
        self.pdf_reader = None
        self.num_pages = 0

    def load(self, file_path: str) -> None:
        """
        PDFファイルを読み込み、PyPDF2を使って解析します

        Args:
            file_path (str): PDFファイルのパス
        """
        with open(file_path, 'rb') as file:
            self.pdf_reader = PyPDF2.PdfReader(file)
            self.num_pages = len(self.pdf_reader.pages)

    def extract_text(self) -> str:
        """
        PDFからすべてのテキストを抽出します

        Returns:
            str: 抽出されたテキスト
        """
        if not self.pdf_reader:
            return ""

        text = ""
        for page_num in range(self.num_pages):
            page = self.pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"

        return text.strip()

    def extract_metadata(self) -> Dict[str, Any]:
        """
        PDFファイルからメタデータを抽出します

        Returns:
            Dict[str, Any]: メタデータのキーと値のペア
        """
        if not self.pdf_reader:
            return {}

        # PDFドキュメント情報を取得
        metadata = {}
        info = self.pdf_reader.metadata

        if info:
            # 標準メタデータフィールドを処理
            for key in info:
                # キー名を正規化
                clean_key = key
                if key.startswith('/'):
                    clean_key = key[1:]

                # メタデータの値を処理（日付形式など）
                value = info[key]
                if isinstance(value, str) and value.startswith('D:'):
                    # PDF日付形式を解析
                    try:
                        dt = datetime.strptime(value[2:16], '%Y%m%d%H%M%S')
                        value = dt.isoformat()
                    except (ValueError, TypeError):
                        pass

                metadata[clean_key] = value

        # 追加のPDF固有メタデータ
        metadata.update({
            "page_count": self.num_pages,
            "is_encrypted": self.pdf_reader.is_encrypted,
        })

        return metadata

    def extract_sections(self) -> List[Dict[str, str]]:
        """
        PDFからセクションを抽出します
        見出しのパターンに基づいてテキストを分割しようとします

        Returns:
            List[Dict[str, str]]: セクションのリスト（タイトルとコンテンツを含む）
        """
        text = self.extract_text()

        # PDFからのセクション抽出は複雑です
        # 見出しを識別するための簡単なヒューリスティックを使用

        # 見出しとして可能性のあるパターン
        heading_patterns = [
            r'^第[一二三四五六七八九十]+章\s+(.*?)$',  # 日本語で「第X章 タイトル」
            r'^第[0-9]+章\s+(.*?)$',                   # 「第N章 タイトル」
            r'^[0-9]+\.\s+(.*?)$',                    # 「N. タイトル」
            r'^[IVX]+\.\s+(.*?)$',                    # ローマ数字を使った見出し
            r'^[A-Z][\.\)]\s+(.*?)$',                 # 「A. タイトル」または「A) タイトル」
            r'^([A-Z][A-Za-z\s]+)$'                   # すべて大文字または先頭が大文字の単語
        ]

        sections = []
        current_title = "はじめに"
        current_content = []

        lines = text.split('\n')

        for line in lines:
            is_heading = False

            # 各行が見出しかどうかをチェック
            for pattern in heading_patterns:
                if re.match(pattern, line.strip()):
                    # 見出しが見つかったら、現在のセクションを保存し、新しいセクションを開始
                    if current_content:
                        sections.append({
                            "title": current_title,
                            "content": '\n'.join(current_content).strip()
                        })

                    current_title = line.strip()
                    current_content = []
                    is_heading = True
                    break

            if not is_heading:
                current_content.append(line)

        # 最後のセクションを追加
        if current_content:
            sections.append({
                "title": current_title,
                "content": '\n'.join(current_content).strip()
            })

        # セクションが識別されなかった場合、デフォルトのセクション分けを使用
        if not sections:
            sections = super().extract_sections()

        return sections
