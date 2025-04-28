import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base import BaseDocumentProcessor


class TEXTProcessor(BaseDocumentProcessor):
    """
    テキストファイル (.txt) を処理するためのプロセッサークラス
    """

    def __init__(self):
        """
        テキストプロセッサーの初期化
        """
        super().__init__()
        self.text_content = ""

    def load(self, file_path: str) -> None:
        """
        テキストファイルを読み込みます

        Args:
            file_path (str): テキストファイルのパス
        """
        try:
            # UTF-8でまず試してみる
            with open(file_path, 'r', encoding='utf-8') as file:
                self.text_content = file.read()
        except UnicodeDecodeError:
            # UTF-8で失敗した場合、Shift-JISで試す
            try:
                with open(file_path, 'r', encoding='shift_jis') as file:
                    self.text_content = file.read()
            except UnicodeDecodeError:
                # それでも失敗した場合、他のエンコーディングを試す
                encodings = ['euc-jp', 'iso-2022-jp', 'cp932']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            self.text_content = file.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # すべてのエンコーディングが失敗した場合
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                        self.text_content = file.read()

    def extract_text(self) -> str:
        """
        テキストコンテンツをそのまま返します

        Returns:
            str: 抽出されたテキスト
        """
        return self.text_content

    def extract_metadata(self) -> Dict[str, Any]:
        """
        テキストファイルのメタデータを抽出します

        Returns:
            Dict[str, Any]: メタデータのキーと値のペア
        """
        # テキストファイルにはメタデータが少ないので、基本的な統計情報を提供
        metadata = {
            "character_count": len(self.text_content),
            "line_count": len(self.text_content.splitlines()),
            "word_count": len(re.findall(r'\b\w+\b', self.text_content)),
        }

        # 文字コードを推定
        try:
            import chardet
            encoding_result = chardet.detect(self.text_content.encode())
            metadata["encoding"] = encoding_result['encoding']
            metadata["encoding_confidence"] = encoding_result['confidence']
        except (ImportError, AttributeError):
            pass

        return metadata

    def extract_sections(self) -> List[Dict[str, str]]:
        """
        テキストからセクションを抽出します
        見出しのパターンを探して文書を分割します

        Returns:
            List[Dict[str, str]]: セクションのリスト（タイトルとコンテンツを含む）
        """
        if not self.text_content:
            return super().extract_sections()

        # 見出しとして可能性のあるパターン
        heading_patterns = [
            r'^#+\s+(.*?)$',                        # Markdownスタイルの見出し
            r'^第[一二三四五六七八九十]+章\s+(.*?)$',  # 日本語で「第X章 タイトル」
            r'^第[0-9]+章\s+(.*?)$',                # 「第N章 タイトル」
            r'^[0-9]+\.\s+(.*?)$',                  # 「N. タイトル」
            r'^[A-Z][\.\)]\s+(.*?)$',               # 「A. タイトル」または「A) タイトル」
            r'^(.+)\n[=\-]{3,}$'                    # アンダーラインスタイルの見出し
        ]

        sections = []
        current_title = "はじめに"
        current_content = []

        lines = self.text_content.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i]
            is_heading = False

            # アンダーラインスタイルの見出しをチェック
            if i < len(lines) - 1 and re.match(r'^[=\-]{3,}$', lines[i + 1]):
                current_title = line
                i += 2  # 見出しとアンダーラインをスキップ

                # 前のセクションを保存
                if current_content:
                    sections.append({
                        "title": "はじめに" if not sections else current_title,
                        "content": '\n'.join(current_content).strip()
                    })

                current_content = []
                is_heading = True
                continue

            # 他の見出しパターンをチェック
            for pattern in heading_patterns:
                if re.match(pattern, line.strip()):
                    # 前のセクションを保存
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

            i += 1

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
