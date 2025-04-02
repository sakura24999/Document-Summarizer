import os
import anthropic
import time
import logging
from typing import Dict, Any, List, Optional
from .prompt_templates import get_summary_prompt

logger = logging.getLogger(__name__)

class ClaudeSummarizer:
    """Claude APIを使用した文書要約クラス"""

    def __init__(self, api_key: str = None):
        """
        初期化

        Args:
            api_key: Anthropic API Key
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def summarize(
        self,
        text: str,
        document_type: str = 'general',
        detail_level: str = 'standard',
        extract_keywords: bool = True
    ) -> Dict[str, Any]:
        """
        文書の要約を生成

        Args:
            text: 要約対象のテキスト
            document_type: 文書タイプ（legal, technical, medical, academic, business, generalなど）
            detail_level: 要約の詳細度（brief, standard, detailed）
            extract_keywords: 重要キーワードを抽出するかどうか

        Returns:
            要約結果の辞書
        """
        start_time = time.time()

        # 文書が長すぎる場合は分割して処理
        if len(text) > 100000:  # 約10万文字で分割
            return self._summarize_long_document(
                text, document_type, detail_level, extract_keywords
            )

        # プロンプトの生成
        prompt = get_summary_prompt(
            document_type=document_type,
            detail_level=detail_level,
            extract_keywords=extract_keywords
        )

        try:
            # Claude APIを呼び出し
            logger.info(f"Calling Claude API for {document_type} document, detail level: {detail_level}")
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                system=prompt["system"],
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt['user']}\n\n文書:\n{text[:100000]}"  # 10万文字まで
                    }
                ]
            )

            # XMLタグで構造化された応答を解析
            summary, keywords = self._parse_response(response.content)

            processing_time = time.time() - start_time

            return {
                "summary": summary,
                "keywords": keywords,
                "detail_level": detail_level,
                "processing_info": {
                    "processing_time_seconds": processing_time,
                    "model": "claude-3-7-sonnet-20250219",
                    "document_length": len(text)
                }
            }

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            raise

    def _summarize_long_document(
        self,
        text: str,
        document_type: str,
        detail_level: str,
        extract_keywords: bool
    ) -> Dict[str, Any]:
        """長い文書を分割して要約"""
        # 文書を適切なチャンクに分割
        chunks = self._split_text(text, chunk_size=80000)  # 約8万文字ずつ

        # 各チャンクを個別に要約
        chunk_summaries = []
        all_keywords = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_result = self.summarize(
                text=chunk,
                document_type=document_type,
                detail_level=detail_level,
                extract_keywords=extract_keywords
            )
            chunk_summaries.append(chunk_result["summary"])
            all_keywords.extend(chunk_result.get("keywords", []))

        # 全ての要約をさらに要約
        combined_summary = "\n\n".join(chunk_summaries)
        final_result = self.summarize(
            text=combined_summary,
            document_type=document_type,
            detail_level=detail_level,
            extract_keywords=False  # 既にキーワードは抽出済み
        )

        # キーワードを頻度でフィルタリング
        unique_keywords = list(set(all_keywords))

        return {
            "summary": final_result["summary"],
            "keywords": unique_keywords[:20],  # 上位20個
            "detail_level": detail_level,
            "processing_info": {
                **final_result["processing_info"],
                "chunks_processed": len(chunks)
            }
        }

    def _split_text(self, text: str, chunk_size: int = 80000) -> List[str]:
        """テキストを適切なチャンクに分割"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # チャンクサイズに近い位置で句点を探す
            end_pos = min(current_pos + chunk_size, len(text))

            # 理想的には文の終わりで分割
            if end_pos < len(text):
                # 句点（。）で終わる位置を探す
                last_period = text.rfind('。', current_pos, end_pos)
                if last_period > current_pos + chunk_size // 2:  # 少なくとも半分以上進んでいる場合
                    end_pos = last_period + 1

            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos

        return chunks

    def _parse_response(self, response_text: str) -> tuple:
        """Claude APIの応答を解析"""
        summary = response_text
        keywords = []

        # キーワード抽出の例（実際の応答形式に合わせて調整が必要）
        if '<keywords>' in response_text and '</keywords>' in response_text:
            try:
                keywords_text = response_text.split('<keywords>')[1].split('</keywords>')[0]
                keywords = [k.strip() for k in keywords_text.split(',')]

                # 要約部分を取得
                summary = response_text.split('</keywords>')[1].strip()
            except Exception as e:
                logger.warning(f"Failed to parse keywords: {str(e)}")

        return summary, keywords
