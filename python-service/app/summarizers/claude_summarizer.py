import os
import json
import logging
import anthropic
from typing import Dict, List, Any, Optional, Tuple

# プロンプトテンプレートをインポート
from app.summarizers.prompt_templates import get_summary_prompt

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeSummarizer:
    """
    Claude APIを使用して文書を要約するクラス
    """

    def __init__(self, api_key=None, model=None):
        """
        初期化

        Args:
            api_key: Anthropic API キー（Noneの場合は環境変数から取得）
            model: 使用するモデル名（Noneの場合はデフォルト値を使用）
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API キーが設定されていません")

        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "4000"))
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", "20000"))  # 文字数での最大チャンクサイズ

        # Anthropicクライアントの初期化（proxiesパラメータなし）
        try:
            # 新しい初期化方法
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except TypeError:
            # 互換性のある初期化方法
            import http_proxy_manager
            self.client = anthropic.Client(api_key=self.api_key)

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        テキストをチャンクに分割

        Args:
            text: 分割するテキスト

        Returns:
            チャンクのリスト
        """
        # 段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            # 段落の長さ（文字数）
            para_length = len(para)

            # 段落が単独でチャンクサイズを超える場合は、さらに分割
            if para_length > self.chunk_size:
                # すでに蓄積した段落があれば、チャンクとして追加
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 長い段落をセンテンス単位で分割
                sentences = para.replace('. ', '.\n').split('\n')
                sub_chunk = []
                sub_length = 0

                for sentence in sentences:
                    sentence_length = len(sentence)
                    if sub_length + sentence_length > self.chunk_size:
                        if sub_chunk:
                            chunks.append(' '.join(sub_chunk))
                        sub_chunk = [sentence]
                        sub_length = sentence_length
                    else:
                        sub_chunk.append(sentence)
                        sub_length += sentence_length

                if sub_chunk:
                    chunks.append(' '.join(sub_chunk))

            # 通常の段落処理
            elif current_length + para_length > self.chunk_size:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length

        # 残りの段落をチャンクとして追加
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        logger.info(f"テキストを {len(chunks)} チャンクに分割しました")
        return chunks

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        レスポンステキストからJSONデータを抽出

        Args:
            response_text: APIからのレスポンステキスト

        Returns:
            抽出されたJSONデータ（辞書）
        """
        try:
            # JSONブロックを探す
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"JSONの解析に失敗しました: {e}")

        # JSON形式でなかった場合、キーワードをXMLタグから抽出
        keywords = []
        if '<keywords>' in response_text and '</keywords>' in response_text:
            start_tag = response_text.find('<keywords>') + len('<keywords>')
            end_tag = response_text.find('</keywords>')
            if start_tag < end_tag:
                keywords_str = response_text[start_tag:end_tag].strip()
                keywords = [k.strip() for k in keywords_str.split(',')]

        # 要約テキストとキーワードの辞書を返す
        return {
            "summary": response_text.replace('<keywords>', '').replace('</keywords>', ''),
            "keywords": keywords
        }

    def _call_claude_api(self, text: str, prompt_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Claude APIを呼び出す

        Args:
            text: 要約するテキスト
            prompt_dict: プロンプト辞書（system, user）

        Returns:
            APIレスポンスから抽出した結果
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=prompt_dict["system"],
                messages=[
                    {"role": "user", "content": f"{prompt_dict['user']}\n\n{text}"}
                ]
            )

            response_text = response.content[0].text
            return self._extract_json_from_response(response_text)

        except Exception as e:
            logger.error(f"Claude API呼び出し中にエラーが発生しました: {e}")
            raise

    def summarize(self, text: str, document_type: str = 'general',
                  detail_level: str = 'standard', extract_keywords: bool = True) -> Dict[str, Any]:
        """
        文書を要約する

        Args:
            text: 要約するテキスト
            document_type: 文書タイプ
            detail_level: 詳細レベル
            extract_keywords: キーワードを抽出するかどうか

        Returns:
            要約結果の辞書
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("要約するテキストが空です")

        logger.info(f"文書要約開始: タイプ={document_type}, 詳細レベル={detail_level}")

        # テキストの長さをチェックしてチャンク処理が必要か判断
        if len(text) <= self.chunk_size:
            # 短いテキストは直接処理
            logger.info("短いテキスト: 直接処理")
            prompt_dict = get_summary_prompt(
                document_type=document_type,
                detail_level=detail_level,
                extract_keywords=extract_keywords
            )

            result = self._call_claude_api(text, prompt_dict)
            result["detail_level"] = detail_level

            return result
        else:
            # 長いテキストはチャンク処理
            logger.info("長いテキスト: チャンク処理開始")
            chunks = self._split_text_into_chunks(text)
            chunk_results = []

            # 各チャンクを順番に処理
            for i, chunk in enumerate(chunks):
                logger.info(f"チャンク {i+1}/{len(chunks)} 処理中...")

                # チャンク用のプロンプトを取得
                prompt_dict = get_summary_prompt(
                    document_type=document_type,
                    detail_level=detail_level,
                    extract_keywords=False,  # 中間チャンクではキーワード抽出しない
                    is_chunk=True,
                    chunk_number=i+1,
                    total_chunks=len(chunks)
                )

                # チャンクを処理
                chunk_result = self._call_claude_api(chunk, prompt_dict)
                chunk_results.append(chunk_result)

            # 全チャンクの要約を結合
            combined_summaries = "\n\n".join([r.get("summary", "") for r in chunk_results])

            # 最終的な要約を生成
            logger.info("全チャンクの統合要約を生成中...")
            final_prompt = get_summary_prompt(
                document_type=document_type,
                detail_level=detail_level,
                extract_keywords=extract_keywords,
                is_final_summary=True
            )

            final_result = self._call_claude_api(combined_summaries, final_prompt)
            final_result["detail_level"] = detail_level

            # 処理情報を追加
            final_result["processing_info"] = {
                "chunks_count": len(chunks),
                "total_length": len(text),
                "chunked_processing": True
            }

            logger.info("チャンク処理完了: 最終要約を生成しました")
            return final_result
