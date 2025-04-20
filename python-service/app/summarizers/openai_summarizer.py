# app/summarizers/openai_summarizer.py
import os
import openai
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class OpenAISummarizer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def summarize(self, text: str, document_type: str = "general", summary_type: str = "standard") -> Tuple[str, List[str]]:
        """
        OpenAIのGPT-3.5 Turboを使って文書を要約し、キーワードを抽出する
        """
        try:
            # 要約の詳細度の設定
            summary_length = {
                "brief": "元の文書の約1/10の長さで",
                "standard": "元の文書の約1/5の長さで",
                "detailed": "元の文書の約1/3の長さで"
            }.get(summary_type, "元の文書の約1/5の長さで")

            # 文書タイプに基づいたプロンプトの調整
            document_type_instruction = {
                "legal": "法律文書として、重要な法的条項や規定に注目して",
                "technical": "技術文書として、技術的な詳細や手順に注目して",
                "medical": "医療文書として、医学的な所見や治療法に注目して",
                "academic": "学術論文として、研究方法や結果、結論に注目して",
                "business": "ビジネス文書として、重要な数字や戦略的な情報に注目して",
                "general": "一般文書として、主要な情報や要点に注目して"
            }.get(document_type, "一般文書として、主要な情報や要点に注目して")

            # プロンプトの構築
            prompt = f"""
            次の文書を{document_type_instruction}要約してください。要約は{summary_length}作成してください。
            また、文書から重要なキーワードを5〜10個抽出してください。

            文書:
            {text}

            出力形式:
            要約: [要約テキスト]
            キーワード: [キーワード1], [キーワード2], [キーワード3], ...
            """

            # OpenAI APIを呼び出し
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは専門文書要約の専門家です。与えられた文書を要約し、重要なキーワードを抽出します。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # レスポンスから要約とキーワードを抽出
            content = response.choices[0].message.content

            # 要約とキーワードを分離
            summary = ""
            keywords = []

            if "要約:" in content and "キーワード:" in content:
                summary_part = content.split("キーワード:")[0].replace("要約:", "").strip()
                keywords_part = content.split("キーワード:")[1].strip()

                summary = summary_part
                keywords = [k.strip() for k in keywords_part.split(",")]
            else:
                # フォーマットが期待通りでない場合のフォールバック処理
                summary = content
                keywords = []

            return summary, keywords

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"文書の要約中にエラーが発生しました: {str(e)}")
