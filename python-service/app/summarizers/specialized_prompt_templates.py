def get_legal_prompt(document_subtype, *args, **kwargs):
    """法務文書のサブタイプ別プロンプト生成

    Args:
        document_subtype: 契約書、判例、法令など
    """
    base_prompt = get_summary_prompt(document_type='legal', *args, **kwargs)

    # 契約書向けの特化プロンプト
    if document_subtype == 'contract':
        base_prompt["system"] += """
契約書の分析において特に注意すべき点:
- 当事者の権利義務関係を明確に区別する
- 責任の所在や制限事項を正確に特定する
- 支払条件や期限を明示する
- 解除・解約条件とその効果を明確にする
- 準拠法・裁判管轄条項の内容を確認する

JSON出力には以下の追加情報を含めてください:
"legal_analysis": {
  "key_clauses": [重要条項の概要],
  "risk_factors": [リスク要因],
  "obligations": [主な義務],
  "termination_conditions": [解約・解除条件]
}
"""

    # 判例向けの特化プロンプト
    elif document_subtype == 'case_law':
        base_prompt["system"] += """
判例分析において特に注意すべき点:
- 事実関係の整理
- 争点の明確化
- 裁判所の判断とその根拠
- 先例との関係性
- 本判決がもつ法的意義

JSON出力には以下の追加情報を含めてください:
"legal_analysis": {
  "facts": [事実関係],
  "issues": [争点],
  "holding": [判旨],
  "reasoning": [理由づけ],
  "significance": [法的意義]
}
"""

    return base_prompt

def get_medical_prompt(document_subtype, *args, **kwargs):
    """医療文書のサブタイプ別プロンプト生成

    Args:
        document_subtype: 診断書、カルテ、医学論文など
    """
    base_prompt = get_summary_prompt(document_type='medical', *args, **kwargs)

    # 診断書向けの特化プロンプト
    if document_subtype == 'diagnosis':
        base_prompt["system"] += """
診断書の分析において特に注意すべき点:
- 診断名と病名の正確な抽出
- 症状の発現時期と経過
- 検査結果と基準値からの逸脱
- 治療計画と処方内容
- 経過観察事項と予後

JSON出力には以下の追加情報を含めてください:
"medical_analysis": {
  "diagnoses": [診断名],
  "symptoms": [主な症状],
  "test_results": [検査結果],
  "treatments": [治療計画],
  "medications": [処方薬]
}
"""

    # 医学論文向けの特化プロンプト
    elif document_subtype == 'research_paper':
        base_prompt["system"] += """
医学論文の分析において特に注意すべき点:
- 研究目的と仮説の明確化
- 研究手法の妥当性
- 統計的有意性を含めた結果の解釈
- 研究の限界点
- 臨床応用の可能性

JSON出力には以下の追加情報を含めてください:
"medical_analysis": {
  "hypothesis": [研究仮説],
  "methodology": [研究手法],
  "findings": [主な発見],
  "limitations": [限界点],
  "clinical_implications": [臨床的意義]
}
"""

    return base_prompt
