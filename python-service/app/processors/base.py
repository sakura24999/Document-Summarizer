import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseDocumentProcessor(ABC):
    """
    文書処理のための基本クラス。
    すべての文書タイプ固有のプロセッサーはこのクラスを継承する必要があります。
    """

    def __init__(self):
        """
        プロセッサーの初期化
        """
        self.content = None
        self.metadata = {}

    @abstractmethod
    def load(self, file_path: str) -> None:
        """
        ファイルを読み込み、内容を解析します

        Args:
            file_path (str): 処理するファイルのパス
        """
        pass

    @abstractmethod
    def extract_text(self) -> str:
        """
        ファイルからテキストを抽出します

        Returns:
            str: 抽出されたテキスト
        """
        pass

    @abstractmethod
    def extract_metadata(self) -> Dict[str, Any]:
        """
        ファイルからメタデータを抽出します

        Returns:
            Dict[str, Any]: メタデータのキーと値のペア
        """
        pass

    def extract_sections(self) -> List[Dict[str, str]]:
        """
        文書からセクションを抽出します

        Returns:
            List[Dict[str, str]]: セクションのリスト（タイトルとコンテンツを含む）
        """
        # デフォルトの実装では、文書全体を1つのセクションとして返します
        text = self.extract_text()
        return [{"title": "Document", "content": text}]

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        ファイルを完全に処理し、すべての抽出された情報を返します

        Args:
            file_path (str): 処理するファイルのパス

        Returns:
            Dict[str, Any]: 抽出されたテキスト、メタデータ、セクションを含む辞書
        """
        self.load(file_path)

        result = {
            "text": self.extract_text(),
            "metadata": self.extract_metadata(),
            "sections": self.extract_sections(),
            "file_info": {
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_size_bytes": os.path.getsize(file_path)
            }
        }

        return result
