import os
import json
import logging
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

        # Anthropicクライアントの初期化（安全に行う）
        try:
            # 新しいバージョンのAnthropicライブラリ用の初期化方法
            import anthropic
            anthropic_version = getattr(anthropic, "__version__", "0.0.0")
            logger.info(f"Anthropic APIバージョン: {anthropic_version}")

            if anthropic_version >= "0.5.0":
                # デバッグ情報を追加
                logger.info("Anthropic 0.5.0以上を使用しています。プロキシなしでクライアントを初期化します。")

                # 明示的にhttpxクライアントを作成してproxies関連の問題を回避
                import httpx
                http_client = httpx.Client(timeout=30.0)

                # proxyが環境変数から設定されている可能性があるため、明示的にAPI_KEYのみ渡す
                self.client = anthropic.Anthropic(
                    api_key=self.api_key,
                    http_client=http_client
                )
                logger.info("新しいAnthropicクライアント初期化が成功しました")
            else:
                # 古いバージョン用
                self.client = anthropic.Client(api_key=self.api_key)
                logger.info("レガシーAnthropicクライアント初期化が成功しました")
        except (ImportError, AttributeError) as e:
            logger.error(f"Anthropicライブラリの初期化に失敗しました: {e}")
            raise ValueError(f"Anthropicライブラリの初期化エラー: {str(e)}")

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
            # APIバージョンによって異なる呼び出し方法をサポート
            import anthropic
            anthropic_version = getattr(anthropic, "__version__", "0.0.0")

            if anthropic_version >= "0.5.0":
                # Anthropic API v0.5.0用のリクエスト形式
                # 古いバージョンのAPI（v0.5.0）では以下の形式を使用
                response = self.client.completions.create(
                    prompt=f"{anthropic.HUMAN_PROMPT} {prompt_dict['user']}\n\n{text}\n\n{anthropic.AI_PROMPT}",
                    model=self.model,
                    max_tokens_to_sample=self.max_tokens,
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                )
                response_text = response.completion
            else:
                # 新しいバージョンのAPI（v1.0以上）用
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
            import traceback
            logger.error(traceback.format_exc())
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
