from typing import Dict, Any, List
from app.summarizers.claude_summarizer import ClaudeSummarizer
from app.analyzers.document_classifier import DocumentClassifier
from app.summarizers.specialized_prompt_templates import get_legal_prompt, get_medical_prompt
from app.summarizers.output_formatters import OutputFormatter

class SpecializedSummarizer:
    """
    専門分野に特化した要約機能を提供するクラス
    """

    def __init__(self):
        """
        初期化
        """
        self.base_summarizer = ClaudeSummarizer()
        self.classifier = DocumentClassifier()

    def summarize(self, text: str, document_type: str = None, document_subtype: str = None,
                 detail_level: str = 'standard', extract_keywords: bool = True) -> Dict[str, Any]:
        """
        文書を専門分野に特化した方法で要約

        Args:
            text: 要約するテキスト
            document_type: 文書タイプ (指定がなければ自動検出)
            document_subtype: 文書サブタイプ (指定がなければ自動検出)
            detail_level: 詳細レベル
            extract_keywords: キーワードを抽出するかどうか

        Returns:
            要約結果の辞書
        """
        # 文書タイプが指定されていなければ自動検出
        if not document_type:
            classification = self.classifier.classify(text)
            document_type = classification['domain']
            document_subtype = classification['subtype']

            # 分類結果をログに記録
            print(f"文書分類結果: {document_type}, サブタイプ: {document_subtype}")

        # 専門分野に応じたプロンプトを取得
        if document_type == 'legal' and document_subtype:
            prompt_dict = get_legal_prompt(document_subtype, detail_level=detail_level,
                                          extract_keywords=extract_keywords)
        elif document_type == 'medical' and document_subtype:
            prompt_dict = get_medical_prompt(document_subtype, detail_level=detail_level,
                                           extract_keywords=extract_keywords)
        else:
            # 基本のプロンプトを使用
            prompt_dict = None

        # 要約処理を実行
        if prompt_dict:
            # 特化プロンプトを使用
            result = self.base_summarizer._call_claude_api(text, prompt_dict)
        else:
            # 標準の要約メソッドを使用
            result = self.base_summarizer.summarize(
                text=text,
                document_type=document_type,
                detail_level=detail_level,
                extract_keywords=extract_keywords
            )

        # 専門分野に応じた出力フォーマットを適用
        formatted_result = OutputFormatter.format_output(result, document_type, document_subtype)

        # 分類情報を追加
        formatted_result['document_info'] = {
            'type': document_type,
            'subtype': document_subtype
        }

        return formatted_result
