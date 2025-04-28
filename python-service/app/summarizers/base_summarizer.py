from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseSummarizer(ABC):
    """
    テキスト要約のための基本クラス。
    すべての要約アルゴリズムはこのクラスを継承する必要があります。
    """

    def __init__(self):
        """
        サマライザーの初期化
        """
        pass

    @abstractmethod
    def summarize(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        テキストを要約します

        Args:
            text (str): 要約する元のテキスト
            **kwargs: 追加のパラメータ（最大長、要約タイプなど）

        Returns:
            Dict[str, Any]: 要約結果を含む辞書
        """
        pass

    @abstractmethod
    def summarize_sections(self, sections: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        セクションごとにテキストを要約します

        Args:
            sections (List[Dict[str, str]]): 要約するセクションのリスト
            **kwargs: 追加のパラメータ（最大長、要約タイプなど）

        Returns:
            Dict[str, Any]: セクションごとの要約結果を含む辞書
        """
        pass

    def preprocess_text(self, text: str) -> str:
        """
        要約前にテキストを前処理します

        Args:
            text (str): 前処理する元のテキスト

        Returns:
            str: 前処理されたテキスト
        """
        # デフォルトでは、単に先頭と末尾の空白を取り除きます
        return text.strip()

    def validate_text(self, text: str) -> bool:
        """
        テキストが要約に有効かどうかを検証します

        Args:
            text (str): 検証するテキスト

        Returns:
            bool: テキストが有効な場合はTrue、そうでない場合はFalse
        """
        # 最小文字数をチェック
        min_length = 50  # 例としての最小長
        if len(text) < min_length:
            return False

        return True
