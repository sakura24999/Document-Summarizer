def get_summary_prompt(document_type='general', detail_level='standard', extract_keywords=True,
                       is_chunk=False, chunk_number=None, total_chunks=None, is_final_summary=False):
    """
    文書タイプと詳細レベルに基づいたプロンプトを生成

    Args:
        document_type: 文書タイプ (legal, technical, medical, academic, business, general)
        detail_level: 要約の詳細レベル (brief, standard, detailed)
        extract_keywords: キーワード抽出を行うかどうか
        is_chunk: 複数チャンクの一部かどうか
        chunk_number: チャンク番号
        total_chunks: 合計チャンク数
        is_final_summary: 最終的な要約を生成するかどうか

    Returns:
        システムプロンプトとユーザープロンプトの辞書
    """
    # 基本となるシステムプロンプト
    base_system_prompt = """あなたは専門文書の要約エキスパートです。与えられた文書を分析し、重要なポイントを抽出し、簡潔かつ包括的な要約を作成してください。"""

    # 文書タイプ別の特化プロンプト
    document_type_prompts = {
        'legal': """法律文書を要約する際は、以下の点に注意してください:
- 法的用語や定義を正確に把握する
- 法的義務や権利を明確に特定する
- 日付や期限の情報を正確に抽出する
- 条件や例外を明示する
- 文書の法的効果を理解して伝える""",

        'technical': """技術文書を要約する際は、以下の点に注意してください:
- 技術的概念や用語を正確に理解する
- 技術仕様や要件を明確に抽出する
- 手順やプロセスを論理的に整理する
- 図表やコードの要点を説明する
- 技術的な制約や注意点を明示する""",

        'medical': """医療文書を要約する際は、以下の点に注意してください:
- 医学用語を正確に理解する
- 診断、症状、治療法を明確に区別する
- 数値データ（投薬量、検査値など）を正確に抽出する
- 時系列情報を整理する
- 患者情報に関する機密性を考慮する""",

        'academic': """学術論文を要約する際は、以下の点に注意してください:
- 研究目的と方法論を明確に抽出する
- 実験設計と結果を正確に要約する
- 統計データや分析結果を適切に解釈する
- 著者の主張と根拠を区別する
- 研究の限界と今後の課題を含める""",

        'business': """ビジネス文書を要約する際は、以下の点に注意してください:
- ビジネス目標や戦略を明確に特定する
- 財務データや市場情報を正確に抽出する
- リスクと機会を区別する
- アクションアイテムや決定事項を強調する
- 期限やマイルストーンを明示する""",

        'general': """一般文書を要約する際は、以下の点に注意してください:
- 文書の主題と目的を特定する
- 主要な論点を抽出する
- 重要な事実やデータを正確に要約する
- 結論や提案事項を明確にする
- 文脈を維持しながら冗長な部分を省略する"""
    }

    # 詳細レベル別の指示
    detail_level_instructions = {
        'brief': "非常に簡潔な要約を作成してください。元の文書の約1/10程度の長さで、最も重要なポイントのみを含めてください。",
        'standard': "バランスの取れた要約を作成してください。元の文書の約1/5程度の長さで、主要な内容をカバーしてください。",
        'detailed': "詳細な要約を作成してください。元の文書の約1/3程度の長さで、重要なポイントと詳細情報を含めてください。"
    }

    # チャンク処理の指示
    chunk_instructions = ""
    if is_chunk:
        chunk_instructions = f"""
これは長い文書の一部（チャンク {chunk_number}/{total_chunks}）です。この部分のみの要約を作成してください。
最終的な要約は別途作成されるため、このチャンクの主要なポイントを抽出することに集中してください。
"""
    elif is_final_summary:
        chunk_instructions = """
これは複数のチャンクに分割された文書の要約をまとめたものです。
これらの要約を統合して、一貫性のある最終的な要約を作成してください。
重複している情報は統合し、文書全体の流れを反映する要約を作成してください。
"""

    # 文書タイプのプロンプトを取得（存在しない場合はgeneralを使用）
    type_prompt = document_type_prompts.get(document_type, document_type_prompts['general'])

    # 詳細レベルの指示を取得
    level_instruction = detail_level_instructions.get(detail_level, detail_level_instructions['standard'])

    # システムプロンプトの組み立て
    system_prompt = f"{base_system_prompt}\n\n{type_prompt}\n\n{level_instruction}"

    # チャンク処理の指示があれば追加
    if chunk_instructions:
        system_prompt += f"\n\n{chunk_instructions}"

    # キーワード抽出の指示を追加
    if extract_keywords and not is_chunk:
        system_prompt += "\n\n要約に加えて、文書から重要なキーワードを最大20個抽出し、カンマ区切りでリストアップしてください。キーワードはXML形式で<keywords>キーワード1, キーワード2, ...</keywords>のように提示してください。"

    # JSON形式で返すように指示（チャンク処理やデータ統合に便利）
    if is_final_summary or not is_chunk:
        system_prompt += """
出力形式:
JSONフォーマットで以下の情報を含めてください：
{
  "summary": "要約テキスト",
  "keywords": ["キーワード1", "キーワード2", "キーワード3", ...]
}
"""

    # ユーザープロンプト
    user_prompt = "以下の文書を要約してください。"

    if is_chunk:
        user_prompt = f"以下の文書（チャンク {chunk_number}/{total_chunks}）を要約してください。"
    elif is_final_summary:
        user_prompt = "以下は複数の要約結果です。これらを統合して最終的な要約を作成してください。"
    elif detail_level == 'brief':
        user_prompt += "要点のみを簡潔に抽出してください。"
    elif detail_level == 'detailed':
        user_prompt += "重要なポイントと詳細な説明を含めてください。"

    return {
        "system": system_prompt,
        "user": user_prompt
    }
