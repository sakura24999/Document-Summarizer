import logging
from typing import Dict, Any, List, Optional
import spacy
from app.summarizers.claude_summarizer import ClaudeSummarizer
from app.summarizers.transformers_summarizer import TransformersSummarizer
from app.analyzers.document_classifier import DocumentClassifier
from app.summarizers.specialized_prompt_templates import get_legal_prompt, get_medical_prompt
from app.summarizers.output_formatters import OutputFormatter
import numpy as np
import re

# 依存関係を安全にインポート
try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    logging.warning("sklearn依存関係のインポートに失敗しました。一部機能が制限されます。")

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMER = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMER = False
    logging.warning("sentence_transformersのインポートに失敗しました。文埋め込み機能が制限されます。")

# ロギング設定
logger = logging.getLogger(__name__)

class EnhancedSpecializedSummarizer:
    """
    NLPライブラリを活用した高度な専門文書要約クラス
    """

    def __init__(self, use_transformers: bool = True):
        """
        初期化

        Args:
            use_transformers: Transformersベースの要約を使用するかどうか
        """
        # 基本サマライザーの初期化
        self.claude_summarizer = ClaudeSummarizer()

        # Transformersベースのサマライザー
        self.transformers_summarizer = None
        if use_transformers:
            try:
                self.transformers_summarizer = TransformersSummarizer()
                logger.info("TransformersSummarizerを初期化しました")
            except Exception as e:
                logger.error(f"TransformersSummarizerの初期化に失敗しました: {e}")
                logger.info("Transformersサマライザーなしで続行します")

        # 文書分類器
        try:
            self.classifier = DocumentClassifier()
            logger.info("DocumentClassifierを初期化しました")
        except Exception as e:
            logger.error(f"DocumentClassifierの初期化に失敗しました: {e}")
            self.classifier = None

        # spaCy NLPモデル
        self.nlp = None
        try:
            self.nlp = spacy.load('ja_core_news_lg')
            logger.info("日本語spaCyモデルを読み込みました")
        except OSError:
            try:
                logger.info("日本語spaCyモデルをダウンロードしています...")
                spacy.cli.download('ja_core_news_lg')
                self.nlp = spacy.load('ja_core_news_lg')
            except Exception as e:
                logger.error(f"spaCyモデルのダウンロードに失敗しました: {e}")
                try:
                    # 小さいモデルを試す
                    logger.info("代替としてja_core_news_mdを試みます")
                    spacy.cli.download('ja_core_news_md')
                    self.nlp = spacy.load('ja_core_news_md')
                except Exception:
                    logger.error("すべてのspaCyモデルの読み込みに失敗しました")

        # 文埋め込みモデル
        self.sentence_model = None
        if HAVE_SENTENCE_TRANSFORMER:
            try:
                self.sentence_model = SentenceTransformer('cl-tohoku/bert-base-japanese-v3')
                logger.info("SentenceTransformerモデルを初期化しました")
            except Exception as e:
                logger.error(f"SentenceTransformerモデルの初期化に失敗しました: {e}")

        # 使用するNLPライブラリの情報をログに記録
        logger.info(f"EnhancedSpecializedSummarizer初期化: Transformers={use_transformers}")

    def summarize(self, text: str, document_type: str = None, document_subtype: str = None,
                 detail_level: str = 'standard', extract_keywords: bool = True,
                 use_claude: bool = True) -> Dict[str, Any]:
        """
        文書を専門分野に特化した方法で要約

        Args:
            text: 要約するテキスト
            document_type: 文書タイプ (指定がなければ自動検出)
            document_subtype: 文書サブタイプ (指定がなければ自動検出)
            detail_level: 詳細レベル
            extract_keywords: キーワードを抽出するかどうか
            use_claude: Claude APIを使用するかどうか

        Returns:
            要約結果の辞書
        """
        # テキストの前処理
        processed_text = self._preprocess_text(text)

        # 文書タイプが指定されていなければ自動検出
        if not document_type or not document_subtype:
            if self.classifier:
                try:
                    classification = self.classifier.classify(processed_text)

                    if not document_type:
                        document_type = classification['domain']

                    if not document_subtype:
                        document_subtype = classification['subtype']

                    # 分類結果をログに記録
                    logger.info(f"文書分類結果: {document_type}, サブタイプ: {document_subtype}")
                    logger.info(f"分類信頼度: {classification.get('confidence', 'N/A')}")
                except Exception as e:
                    logger.error(f"文書分類に失敗しました: {e}")
                    # デフォルト値を設定
                    document_type = document_type or 'general'
                    document_subtype = document_subtype or None
            else:
                # 分類器が利用できない場合はデフォルト値を使用
                document_type = document_type or 'general'
                document_subtype = document_subtype or None

        # NLPライブラリを使用した事前解析
        analysis_result = {}
        if self.nlp:
            try:
                analysis_result = self._analyze_with_nlp(processed_text, document_type, document_subtype)
            except Exception as e:
                logger.error(f"NLP解析に失敗しました: {e}")

        # 専門分野に応じたキーワード抽出
        keywords = []
        if extract_keywords:
            try:
                keywords = self._extract_domain_specific_keywords(processed_text, document_type)
            except Exception as e:
                logger.error(f"キーワード抽出に失敗しました: {e}")

        # 要約の生成
        summary = ""
        if use_claude:
            try:
                # 専門分野に応じたプロンプトを取得
                if document_type == 'legal' and document_subtype:
                    prompt_dict = get_legal_prompt(document_subtype, detail_level=detail_level,
                                                extract_keywords=False)  # キーワードは既に抽出済み
                elif document_type == 'medical' and document_subtype:
                    prompt_dict = get_medical_prompt(document_subtype, detail_level=detail_level,
                                                extract_keywords=False)
                else:
                    # 基本のプロンプトを使用
                    prompt_dict = None

                # Claude APIを使用した要約
                if prompt_dict:
                    # 特化プロンプトを使用
                    result = self.claude_summarizer._call_claude_api(processed_text, prompt_dict)
                    summary = result.get("summary", "")
                else:
                    # 標準の要約メソッドを使用
                    result = self.claude_summarizer.summarize(
                        text=processed_text,
                        document_type=document_type,
                        detail_level=detail_level,
                        extract_keywords=False
                    )
                    summary = result.get("summary", "")
            except Exception as e:
                logger.error(f"Claude APIを使用した要約に失敗しました: {e}")
                # フォールバックとしてTransformersを使用
                use_claude = False

        # Transformersベースの要約を使用
        if not use_claude:
            if self.transformers_summarizer:
                try:
                    result = self.transformers_summarizer.summarize(
                        text=processed_text,
                        document_type=document_type,
                        detail_level=detail_level,
                        extract_keywords=False
                    )
                    summary = result.get("summary", "")
                except Exception as e:
                    logger.error(f"Transformersサマライザーを使用した要約に失敗しました: {e}")
                    # フォールバックとして抽出型要約を実行
                    summary = self._extractive_summarization(processed_text, detail_level)
            else:
                # フォールバックとして抽出型要約を実行
                summary = self._extractive_summarization(processed_text, detail_level)

        # 最終的な結果の構築
        final_result = {
            "summary": summary,
            "keywords": keywords,
            "document_info": {
                "type": document_type,
                "subtype": document_subtype
            }
        }

        # NLP解析結果を追加
        if analysis_result:
            if document_type == 'legal':
                final_result['legal_analysis'] = analysis_result
            elif document_type == 'medical':
                final_result['medical_analysis'] = analysis_result
            elif document_type == 'technical':
                final_result['technical_analysis'] = analysis_result
            else:
                final_result['detailed_analysis'] = analysis_result

        return final_result

    def _preprocess_text(self, text: str) -> str:
        """
        テキストの前処理

        Args:
            text: 前処理するテキスト

        Returns:
            前処理されたテキスト
        """
        # 空白行の正規化
        processed = '\n'.join([line for line in text.split('\n') if line.strip()])

        # 文字コードの正規化
        processed = processed.replace('\u3000', ' ')  # 全角スペースを半角に

        # 改行の正規化
        processed = processed.replace('\r\n', '\n')

        return processed

    def _analyze_with_nlp(self, text: str, document_type: str, document_subtype: str) -> Dict[str, Any]:
        """
        NLPライブラリを使用した文書解析

        Args:
            text: 解析するテキスト
            document_type: 文書タイプ
            document_subtype: 文書サブタイプ

        Returns:
            解析結果
        """
        if not self.nlp:
            return {}

        # spaCyによる解析（メモリ制限のため長すぎるテキストは切り詰め）
        max_text_length = min(len(text), 50000)
        doc = self.nlp(text[:max_text_length])

        # 文書タイプに応じた解析
        if document_type == 'legal':
            return self._analyze_legal_document(doc, document_subtype)
        elif document_type == 'medical':
            return self._analyze_medical_document(doc, document_subtype)
        elif document_type == 'technical':
            return self._analyze_technical_document(doc)
        else:
            return self._analyze_general_document(doc)

    def _analyze_legal_document(self, doc, subtype: str) -> Dict[str, Any]:
        """
        法律文書の解析

        Args:
            doc: spaCyのドキュメントオブジェクト
            subtype: 法律文書のサブタイプ

        Returns:
            解析結果
        """
        result = {}

        # 固有表現の抽出
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []

            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)

        # 当事者の特定
        parties = []
        party_patterns = ['当事者', '甲', '乙', '丙', '契約者', '依頼者', '委託者', '受託者']

        for token in doc:
            if token.text in party_patterns:
                # 当事者を特定する文脈を探す
                for child in token.children:
                    if child.dep_ == 'compound' and child.pos_ == 'NOUN':
                        parties.append(f"{child.text}{token.text}")

        # 重要な日付の抽出
        dates = entities.get('DATE', [])

        # 契約書の場合
        if subtype == 'contract':
            # 条項の抽出
            clauses = []
            clause_pattern = re.compile(r'第(\d+|[一二三四五六七八九十]+)条')
            clause_text = doc.text

            for match in clause_pattern.finditer(clause_text):
                # 条項の見出しを抽出
                clause_start = match.start()
                next_line_end = clause_text.find('\n', clause_start)

                if next_line_end > 0:
                    clause_heading = clause_text[clause_start:next_line_end].strip()
                else:
                    clause_heading = clause_text[clause_start:clause_start+50].strip()

                clauses.append(clause_heading)

            result = {
                'parties': parties,
                'dates': dates,
                'key_clauses': clauses[:10]  # 最初の10条項のみ
            }

        # 判例の場合
        elif subtype == 'case_law':
            # 事実関係と判断部分を区別
            facts = []
            holdings = []

            for sent in doc.sents:
                sent_text = sent.text.strip()

                if any(word in sent_text for word in ['事実', '経緯']):
                    facts.append(sent_text)
                elif any(word in sent_text for word in ['判断', '判示', '判決']):
                    holdings.append(sent_text)

            result = {
                'parties': parties,
                'dates': dates,
                'facts': facts[:5],  # 最初の5件
                'holdings': holdings[:5]  # 最初の5件
            }

        return result

    def _analyze_medical_document(self, doc, subtype: str) -> Dict[str, Any]:
        """
        医療文書の解析

        Args:
            doc: spaCyのドキュメントオブジェクト
            subtype: 医療文書のサブタイプ

        Returns:
            解析結果
        """
        # 疾患・症状の抽出
        conditions = []
        condition_patterns = ['症', '病', '炎', '障害', '症候群', '疾患']

        for token in doc:
            if any(pattern in token.text for pattern in condition_patterns):
                conditions.append(token.text)

        # 治療法の抽出
        treatments = []
        treatment_patterns = ['療法', '治療', '手術', '投与', 'mg', '処方']

        for token in doc:
            if any(pattern in token.text for pattern in treatment_patterns):
                # 治療法の文脈を探す
                treatment_term = token.text
                for child in token.children:
                    if child.dep_ == 'compound' and child.pos_ == 'NOUN':
                        treatment_term = f"{child.text}{treatment_term}"

                treatments.append(treatment_term)

        # 重要な数値の抽出
        measurements = []
        measurement_pattern = re.compile(r'\d+(\.\d+)?\s*(mg|ml|g|kg|cc|mmHg|mm)')

        for match in measurement_pattern.finditer(doc.text):
            measurements.append(match.group())

        # 診断書の場合
        if subtype == 'diagnosis':
            result = {
                'diagnoses': list(set(conditions)),
                'treatments': list(set(treatments)),
                'measurements': list(set(measurements))
            }

        # 研究論文の場合
        elif subtype == 'research_paper':
            # 研究方法と結果を区別
            methods = []
            results = []

            for sent in doc.sents:
                sent_text = sent.text.strip()

                if any(word in sent_text for word in ['方法', '手法', '手続き']):
                    methods.append(sent_text)
                elif any(word in sent_text for word in ['結果', '有意', 'p値']):
                    results.append(sent_text)

            result = {
                'conditions': list(set(conditions)),
                'methods': methods[:3],  # 最初の3件
                'results': results[:3],  # 最初の3件
                'measurements': list(set(measurements))
            }
        else:
            result = {
                'conditions': list(set(conditions)),
                'treatments': list(set(treatments))
            }

        return result

    def _analyze_technical_document(self, doc) -> Dict[str, Any]:
        """
        技術文書の解析

        Args:
            doc: spaCyのドキュメントオブジェクト

        Returns:
            解析結果
        """
        # 技術用語の抽出
        technical_terms = []
        for token in doc:
            if token.pos_ == 'NOUN' and not token.is_stop and len(token.text) > 1:
                # 複合名詞を抽出
                compound_term = token.text
                for child in token.children:
                    if child.dep_ == 'compound' and child.pos_ == 'NOUN':
                        compound_term = f"{child.text}{compound_term}"

                technical_terms.append(compound_term)

        # 数値仕様の抽出
        specifications = []
        spec_pattern = re.compile(r'\d+(\.\d+)?\s*(Hz|MHz|GHz|KB|MB|GB|TB|mm|cm|m|kg|V|W)')

        for match in spec_pattern.finditer(doc.text):
            specifications.append(match.group())

        # 手順やステップの抽出
        steps = []
        step_patterns = ['ステップ', '手順', '手続き', '工程', '①', '②', '③', '1\\.', '2\\.', '3\\.']

        for pattern in step_patterns:
            positions = [m.start() for m in re.finditer(pattern, doc.text)]
            for pos in positions:
                # ステップの内容を抽出
                line_end = doc.text.find('\n', pos)
                if line_end > 0:
                    step = doc.text[pos:line_end].strip()
                else:
                    step = doc.text[pos:pos+100].strip()

                steps.append(step)

        return {
            'technical_terms': list(set(technical_terms))[:20],  # 最大20個
            'specifications': list(set(specifications)),
            'procedures': steps[:10]  # 最初の10ステップのみ
        }

    def _analyze_general_document(self, doc) -> Dict[str, Any]:
        """
        一般文書の解析

        Args:
            doc: spaCyのドキュメントオブジェクト

        Returns:
            解析結果
        """
        # 固有表現の抽出
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []

            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)

        # 重要な文の抽出
        important_sentences = []

        # 文ベクトルの計算
        sentences = [sent.text for sent in doc.sents]

        if len(sentences) >= 3 and self.sentence_model and HAVE_SKLEARN:
            try:
                # 埋め込みの計算
                embeddings = self.sentence_model.encode(sentences)

                # 文書の中心ベクトルを計算
                centroid = np.mean(embeddings, axis=0)

                # 各文の中心ベクトルとの類似度を計算
                similarities = cosine_similarity([centroid], embeddings)[0]

                # 類似度が高い順にインデックスを取得
                top_indices = similarities.argsort()[-5:][::-1]  # 上位5文

                # 元の順序でソート
                top_indices = sorted(top_indices)

                # 重要な文を抽出
                important_sentences = [sentences[i] for i in top_indices]
            except Exception as e:
                logger.error(f"文ベクトル計算中にエラーが発生しました: {e}")
                # フォールバックとして先頭の文を重要とみなす
                important_sentences = sentences[:5]
        else:
            # 埋め込みが使えない場合は単純に先頭の文を使用
            important_sentences = sentences[:5]

        return {
            'entities': entities,
            'important_sentences': important_sentences
        }

    def _extract_domain_specific_keywords(self, text: str, document_type: str, max_keywords: int = 20) -> List[str]:
        """
        専門分野に応じたキーワード抽出

        Args:
            text: キーワードを抽出するテキスト
            document_type: 文書タイプ
            max_keywords: 抽出するキーワードの最大数

        Returns:
            キーワードのリスト
        """
        if not self.nlp:
            return []

        # spaCyを使用してテキストを解析（メモリ制限のため長すぎるテキストは切り詰め）
        max_text_length = min(len(text), 50000)
        doc = self.nlp(text[:max_text_length])

        # 基本的なキーワード候補を抽出（名詞、固有名詞など）
        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 1:
                # 複合名詞を抽出
                compound = token.text
                for child in token.children:
                    if child.dep_ == 'compound' and child.pos_ == 'NOUN':
                        compound = f"{child.text}{compound}"

                keywords.append(compound)

        # 専門分野に応じた重要語リスト
        domain_terms = []

        if document_type == 'legal':
            domain_terms = [
                '条項', '契約', '当事者', '義務', '権利', '違反', '解除', '損害賠償',
                '紛争解決', '準拠法', '裁判所', '判決', '法令', '訴訟'
            ]
        elif document_type == 'medical':
            domain_terms = [
                '症状', '診断', '治療', '投薬', '手術', '検査', '処方', '副作用',
                '予後', '臨床', '病理', '患者', '医療', '疾患'
            ]
        elif document_type == 'technical':
            domain_terms = [
                '技術', '仕様', '設計', '実装', '効率', '性能', '機能', 'システム',
                '構造', '手順', '方法', '装置', '測定', 'データ'
            ]

        # 単語の頻度を計算
        word_freq = {}
        for word in keywords:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

        # 専門用語にボーナススコアを付与
        for word, freq in word_freq.items():
            if any(term in word for term in domain_terms):
                word_freq[word] = freq * 1.5  # 重要度を50%増加

        # 重要度順にソート
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # 上位のキーワードを返す
        return [word for word, _ in sorted_keywords[:max_keywords]]

    def _extractive_summarization(self, text: str, detail_level: str) -> str:
        """
        NLPライブラリを使用した抽出型要約

        Args:
            text: 要約するテキスト
            detail_level: 詳細レベル

        Returns:
            抽出型要約
        """
        if not self.nlp:
            # 単純な要約（先頭と末尾の部分を使用）
            paragraphs = text.split('\n\n')
            if len(paragraphs) <= 3:
                return text

            if detail_level == 'brief':
                return paragraphs[0]
            elif detail_level == 'detailed':
                return '\n\n'.join(paragraphs[:3] + paragraphs[-2:])
            else:  # standard
                return '\n\n'.join(paragraphs[:2] + paragraphs[-1:])

        # テキストを文に分割
        doc = self.nlp(text[:50000])
        sentences = [sent.text for sent in doc.sents]

        if len(sentences) <= 3:
            return text

        # 詳細レベルに応じた文の数を決定
        if detail_level == 'brief':
            num_sentences = min(5, max(3, len(sentences) // 10))
        elif detail_level == 'detailed':
            num_sentences = min(20, max(10, len(sentences) // 3))
        else:  # standard
            num_sentences = min(10, max(5, len(sentences) // 5))

        # 埋め込みベースの要約が可能か確認
        if self.sentence_model and HAVE_SKLEARN:
            try:
                # センテンスベクトルの計算
                sentence_embeddings = self.sentence_model.encode(sentences)

                # 文書の中心ベクトルを計算
                centroid = np.mean(sentence_embeddings, axis=0)

                # 各文の中心ベクトルとの類似度を計算
                similarities = cosine_similarity([centroid], sentence_embeddings)[0]

                # 類似度が高い順に文を選択（オリジナルの順序を維持）
                ranked_indices = similarities.argsort()[-num_sentences * 2:][::-1]  # 候補を多めに取得
                selected_indices = sorted(ranked_indices[:num_sentences])

                # 選択された文を結合
                summary = ' '.join([sentences[i] for i in selected_indices])
                return summary
            except Exception as e:
                logger.error(f"埋め込みベースの要約に失敗しました: {e}")
                # フォールバック処理へ

        # 埋め込みが使えない場合のフォールバック
        # 簡易的な重要度計算（文書の冒頭と末尾の文は重要である可能性が高い）
        selected_indices = []

        # 冒頭の文を追加
        head_count = max(1, num_sentences // 3)
        selected_indices.extend(range(min(head_count, len(sentences))))

        # 末尾の文を追加
        tail_count = max(1, num_sentences // 3)
        tail_start = max(head_count, len(sentences) - tail_count)
        selected_indices.extend(range(tail_start, len(sentences)))

                        # 残りの文を均等に選択
        remaining_count = num_sentences - len(selected_indices)
        if remaining_count > 0 and head_count < tail_start:
            step = (tail_start - head_count) // (remaining_count + 1)
            if step > 0:
                for i in range(1, remaining_count + 1):
                    selected_indices.append(head_count + i * step)
            else:
                # ステップが計算できない場合は中間部分から適当に選択
                middle_indices = list(range(head_count, tail_start))
                if middle_indices:
                    selected_count = min(remaining_count, len(middle_indices))
                    selected_middle = np.random.choice(middle_indices, selected_count, replace=False)
                    selected_indices.extend(selected_middle)

        # インデックスをソート
        selected_indices = sorted(list(set(selected_indices)))

        # 選択された文を結合
        summary = ' '.join([sentences[i] for i in selected_indices])
        return summary
