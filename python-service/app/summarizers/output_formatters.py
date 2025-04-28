from typing import Dict, Any, List

class OutputFormatter:
    """
    要約結果を専門分野に応じて最適な形式に整形するクラス
    """

    @staticmethod
    def format_legal_output(summary_result: Dict[str, Any], subtype: str = None) -> Dict[str, Any]:
        """
        法務文書の要約結果を整形

        Args:
            summary_result: 基本要約結果
            subtype: 法務文書のサブタイプ

        Returns:
            整形された出力辞書
        """
        # 基本的な要約結果をコピー
        output = summary_result.copy()

        # legal_analysis がなければ、ベースの要約から生成
        if 'legal_analysis' not in output:
            output['legal_analysis'] = {}

            # 契約書の場合
            if subtype == 'contract':
                # キーワードから主要条項を推測
                keywords = output.get('keywords', [])
                output['legal_analysis']['key_clauses'] = [
                    k for k in keywords if any(term in k.lower() for term in
                    ['条項', '契約', '義務', '権利', '責任', '期間', '解約'])
                ]

            # 判例の場合
            elif subtype == 'case_law':
                # 要約からの重要な事実関係や判旨を抽出する処理
                summary = output.get('summary', '')
                # 簡易的な事実関係と判旨の抽出（本来はより精緻な処理が必要）
                facts = []
                holding = []

                lines = summary.split('\n')
                for line in lines:
                    if '事実' in line or '経緯' in line:
                        facts.append(line)
                    elif '判断' in line or '判示' in line:
                        holding.append(line)

                output['legal_analysis']['facts'] = facts
                output['legal_analysis']['holding'] = holding

        return output

    @staticmethod
    def format_medical_output(summary_result: Dict[str, Any], subtype: str = None) -> Dict[str, Any]:
        """
        医療文書の要約結果を整形

        Args:
            summary_result: 基本要約結果
            subtype: 医療文書のサブタイプ

        Returns:
            整形された出力辞書
        """
        # 基本的な要約結果をコピー
        output = summary_result.copy()

        # medical_analysis がなければ、ベースの要約から生成
        if 'medical_analysis' not in output:
            output['medical_analysis'] = {}

            # 診断書の場合
            if subtype == 'diagnosis':
                # キーワードから診断名や治療を推測
                keywords = output.get('keywords', [])

                # 診断名と思われるキーワードを抽出
                diagnoses = [
                    k for k in keywords if any(term in k.lower() for term in
                    ['症', '病', '炎', '障害', '症候群', '疾患'])
                ]

                # 治療法と思われるキーワードを抽出
                treatments = [
                    k for k in keywords if any(term in k.lower() for term in
                    ['療法', '手術', '投与', '治療', 'リハビリ'])
                ]

                output['medical_analysis']['diagnoses'] = diagnoses
                output['medical_analysis']['treatments'] = treatments

        return output

    @staticmethod
    def format_output(summary_result: Dict[str, Any], domain: str, subtype: str = None) -> Dict[str, Any]:
        """
        専門分野に応じた出力形式に整形

        Args:
            summary_result: 基本要約結果
            domain: 専門分野
            subtype: 文書のサブタイプ

        Returns:
            整形された出力辞書
        """
        if domain == 'legal':
            return OutputFormatter.format_legal_output(summary_result, subtype)
        elif domain == 'medical':
            return OutputFormatter.format_medical_output(summary_result, subtype)
        else:
            # 一般的な出力形式を返す
            return summary_result
