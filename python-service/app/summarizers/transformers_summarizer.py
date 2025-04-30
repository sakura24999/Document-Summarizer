import os
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModel
)
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
from sentence_transformers import SentenceTransformer

# NLTKのダウンロードを確認
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformersSummarizer:
    """
    Transformersライブラリを使用して高度な文書要約を行うクラス
    """

    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-v3",
                 summarization_model: str = "google/mt5-small"):
        """
        初期化

        Args:
            model_name: 埋め込みモデルの名前
            summarization_model: 要約モデルの名前
        """
        # 日本語spaCyモデルの読み込み
        try:
            self.nlp = spacy.load("ja_core_news_lg")
            logger.info("日本語spaCyモデルを読み込みました")
        except OSError:
            logger.info("日本語spaCyモデルをダウンロードしています...")
            spacy.cli.download("ja_core_news_lg")
            self.nlp = spacy.load("ja_core_news_lg")

        # 文埋め込みモデルの読み込み
        self.sentence_model = SentenceTransformer(model_name)
        logger.info(f"文埋め込みモデル {model_name} を読み込みました")

        # 要約モデルの準備
        self.tokenizer = AutoTokenizer.from_pretrained(summarization_model)
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model)
        logger.info(f"要約モデル {summarization_model} を読み込みました")

        # 要約パイプラインの作成
        self.summarizer = pipeline(
            "summarization",
            model=self.summarization_model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        # チャンク処理のパラメータ
        self.max_chunk_length = 1024  # トークナイザに送信できる最大長
        self.chunk_overlap = 100      # チャンク間の重複トークン数

        logger.info("TransformersSummarizerの初期化が完了しました")

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

        # チャンク分割が必要かどうかを判断
        if len(text) > self.max_chunk_length:
            # 長いテキストはチャンク処理
            summary = self._summarize_long_text(text, document_type, detail_level)
        else:
            # 短いテキストは直接処理
            summary = self._summarize_text(text, document_type, detail_level)

        # キーワード抽出
        keywords = []
        if extract_keywords:
            keywords = self._extract_keywords(text)

        # 文書タイプに応じた特別な解析
        specialized_analysis = self._analyze_by_document_type(text, document_type)

        result = {
            "summary": summary,
            "keywords": keywords,
            "detail_level": detail_level
        }

        # 特別な解析結果がある場合は追加
        if specialized_analysis:
            result.update(specialized_analysis)

        return result

    def _summarize_text(self, text: str, document_type: str, detail_level: str) -> str:
        """
        短いテキストを直接要約

        Args:
            text: 要約するテキスト
            document_type: 文書タイプ
            detail_level: 詳細レベル

        Returns:
            要約テキスト
        """
        # 詳細レベルに応じた出力長の設定
        max_length = self._get_max_length(len(text), detail_level)
        min_length = max(30, max_length // 2)

        # 要約プロンプトの調整（文書タイプに応じた指示を追加）
        prompt_prefix = self._get_type_specific_prompt(document_type)
        augmented_text = f"{prompt_prefix}\n\n{text}"

        # モデルによる要約の生成
        try:
            summary_output = self.summarizer(
                augmented_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            summary = summary_output[0]['summary_text']
            return summary

        except Exception as e:
            logger.error(f"要約生成中にエラーが発生しました: {e}")
            # フォールバックとして抽出型要約を実行
            return self._extractive_summarization(text, detail_level)

    def _summarize_long_text(self, text: str, document_type: str, detail_level: str) -> str:
        """
        長いテキストをチャンクに分割して要約

        Args:
            text: 要約するテキスト
            document_type: 文書タイプ
            detail_level: 詳細レベル

        Returns:
            要約テキスト
        """
        logger.info("長いテキストをチャンク処理します")

        # テキストをチャンクに分割
        chunks = self._split_into_chunks(text)
        logger.info(f"{len(chunks)}個のチャンクに分割しました")

        # 各チャンクを要約
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"チャンク {i+1}/{len(chunks)} を処理中...")
            chunk_summary = self._summarize_text(chunk, document_type, "brief")
            chunk_summaries.append(chunk_summary)

        # チャンク要約を結合
        intermediate_summary = "\n\n".join(chunk_summaries)
        logger.info("中間要約を生成しました")

        # 最終要約を生成
        final_summary = self._summarize_text(intermediate_summary, document_type, detail_level)
        logger.info("最終要約を生成しました")

        return final_summary

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        テキストを意味のある単位でチャンクに分割

        Args:
            text: 分割するテキスト

        Returns:
            チャンクのリスト
        """
        # spaCyを使用して文を抽出
        doc = self.nlp(text[:50000])  # メモリ制限のため長すぎるテキストは切り詰め
        sentences = [sent.text for sent in doc.sents]

        # 文が少ない場合は段落で分割
        if len(sentences) < 5:
            chunks = text.split('\n\n')
            if len(chunks) > 1:
                return chunks

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # センテンスのトークン数を概算
            sentence_length = len(self.tokenizer.tokenize(sentence))

            if current_length + sentence_length > self.max_chunk_length:
                # 現在のチャンクが最大長を超える場合
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # 単一の文が最大長を超える場合は分割
                    logger.warning(f"長すぎる文を検出: {len(sentence)} 文字")
                    sentence_parts = self._split_long_sentence(sentence)
                    for part in sentence_parts:
                        chunks.append(part)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # 最後のチャンクを追加
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        長い文を適切に分割

        Args:
            sentence: 分割する文

        Returns:
            分割された文のリスト
        """
        # 句読点で分割
        parts = re.split(r'[、。,.;:；：]', sentence)

        # 意味のある単位に再構成
        result = []
        current_part = []
        current_length = 0

        for part in parts:
            part_length = len(self.tokenizer.tokenize(part))

            if current_length + part_length > self.max_chunk_length // 2:
                if current_part:
                    result.append(''.join(current_part))
                    current_part = [part]
                    current_length = part_length
                else:
                    # 単一の部分が長すぎる場合は文字数で分割
                    result.append(part[:self.max_chunk_length // 2])
                    result.append(part[self.max_chunk_length // 2:])
            else:
                current_part.append(part)
                current_length += part_length

        # 最後の部分を追加
        if current_part:
            result.append(''.join(current_part))

        return result

    def _extractive_summarization(self, text: str, detail_level: str) -> str:
        """
        抽出型要約によるバックアップメソッド

        Args:
            text: 要約するテキスト
            detail_level: 詳細レベル

        Returns:
            要約テキスト
        """
        # 文分割
        sentences = [sent.text for sent in self.nlp(text).sents]

        if len(sentences) <= 3:
            return text

        # センテンスベクトルの計算
        sentence_embeddings = self.sentence_model.encode(sentences)

        # 文書全体の中心ベクトルを計算
        centroid = np.mean(sentence_embeddings, axis=0)

        # 各文の中心ベクトルとの類似度を計算
        similarities = cosine_similarity([centroid], sentence_embeddings)[0]

        # 詳細レベルに応じた文の数を決定
        num_sentences = self._get_sentence_count(len(sentences), detail_level)

        # 類似度が高い順に文を選択（オリジナルの順序を維持）
        ranked_indices = similarities.argsort()[::-1][:num_sentences * 2]  # 候補を多めに取得
        selected_indices = sorted(ranked_indices[:num_sentences])

        # 選択された文を結合
        summary = ' '.join([sentences[i] for i in selected_indices])

        return summary

    def _get_max_length(self, text_length: int, detail_level: str) -> int:
        """
        詳細レベルに応じた最大出力長を決定

        Args:
            text_length: 元のテキストの長さ
            detail_level: 詳細レベル

        Returns:
            最大トークン数
        """
        if detail_level == 'brief':
            return min(200, max(50, text_length // 10))
        elif detail_level == 'detailed':
            return min(500, max(150, text_length // 3))
        else:  # standard
            return min(350, max(100, text_length // 5))

    def _get_sentence_count(self, sentence_count: int, detail_level: str) -> int:
        """
        詳細レベルに応じた抽出する文の数を決定

        Args:
            sentence_count: 元の文の数
            detail_level: 詳細レベル

        Returns:
            抽出する文の数
        """
        if detail_level == 'brief':
            return min(5, max(3, sentence_count // 10))
        elif detail_level == 'detailed':
            return min(20, max(10, sentence_count // 3))
        else:  # standard
            return min(10, max(5, sentence_count // 5))

    def _get_type_specific_prompt(self, document_type: str) -> str:
        """
        文書タイプに応じたプロンプトプレフィックスを取得

        Args:
            document_type: 文書タイプ

        Returns:
            プロンプトプレフィックス
        """
        prompts = {
            'legal': "以下は法律文書です。法的な用語、権利義務関係、条件などを正確に要約してください。",
            'medical': "以下は医療文書です。診断、治療法、医学的所見を正確に要約してください。",
            'technical': "以下は技術文書です。技術的な概念、手順、仕様を正確に要約してください。",
            'academic': "以下は学術論文です。研究目的、方法論、結果、結論を要約してください。",
            'business': "以下はビジネス文書です。ビジネス目標、戦略、財務データ、アクション項目を要約してください。",
            'general': "以下の文書を要約してください。"
        }

        return prompts.get(document_type, prompts['general'])

    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        テキストから重要なキーワードを抽出

        Args:
            text: キーワードを抽出するテキスト
            max_keywords: 抽出するキーワードの最大数

        Returns:
            キーワードのリスト
        """
        # spaCyを使用してテキストを解析
        doc = self.nlp(text[:50000])  # メモリ制限のため長すぎるテキストは切り詰め

        # 名詞、固有名詞、形容詞を抽出
        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and len(token.text) > 1:
                keywords.append(token.text)

        # 単語の頻度を計算
        word_freq = {}
        for word in keywords:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

        # TF-IDF的な重み付け（簡易版）
        # 文書全体のトークン数
        total_tokens = len(doc)

        # 単語の重要度を計算
        word_importance = {}
        for word, freq in word_freq.items():
            # 単語の頻度を文書の長さで調整
            normalized_freq = freq / total_tokens

            # 一般的すぎる単語や稀すぎる単語にペナルティ
            if freq <= 1 or freq > total_tokens / 10:
                word_importance[word] = normalized_freq * 0.5
            else:
                word_importance[word] = normalized_freq

        # 重要度順にソート
        sorted_keywords = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)

        # 上位のキーワードを抽出
        return [word for word, _ in sorted_keywords[:max_keywords]]

    def _analyze_by_document_type(self, text: str, document_type: str) -> Dict[str, Any]:
        """
        文書タイプに応じた特別な解析を実行

        Args:
            text: 解析するテキスト
            document_type: 文書タイプ

        Returns:
            解析結果の辞書
        """
        result = {}

        # 法律文書の特別解析
        if document_type == 'legal':
            result['legal_analysis'] = self._analyze_legal_document(text)

        # 医療文書の特別解析
        elif document_type == 'medical':
            result['medical_analysis'] = self._analyze_medical_document(text)

        # 技術文書の特別解析
        elif document_type == 'technical':
            result['technical_analysis'] = self._analyze_technical_document(text)

        return result

    def _analyze_legal_document(self, text: str) -> Dict[str, Any]:
        """
        法律文書の特別解析

        Args:
            text: 解析するテキスト

        Returns:
            法律分析結果
        """
        doc = self.nlp(text[:50000])

        # 当事者の抽出
        parties = []
        party_patterns = ['当事者', '甲', '乙', '丙', '契約者', '依頼者', '委託者', '受託者']
        for ent in doc.ents:
            if ent.label_ == 'PERSON' or ent.label_ == 'ORG':
                parties.append(ent.text)

        for token in doc:
            if token.text in party_patterns:
                # 当事者を特定する文脈を探す
                for child in token.children:
                    if child.dep_ == 'compound' and child.pos_ == 'NOUN':
                        parties.append(f"{child.text}{token.text}")

        # 重要な日付の抽出
        dates = []
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                dates.append(ent.text)

        # 条項の抽出
        clauses = []
        clause_pattern = re.compile(r'第(\d+|[一二三四五六七八九十]+)条')
        for match in clause_pattern.finditer(text):
            # 条項の見出しを抽出
            clause_start = match.start()
            next_line_end = text.find('\n', clause_start)
            if next_line_end > 0:
                clause_heading = text[clause_start:next_line_end].strip()
            else:
                clause_heading = text[clause_start:clause_start+50].strip()

            clauses.append(clause_heading)

        return {
            'parties': list(set(parties)),
            'dates': list(set(dates)),
            'key_clauses': clauses[:10]  # 最初の10条項のみ
        }

    def _analyze_medical_document(self, text: str) -> Dict[str, Any]:
        """
        医療文書の特別解析

        Args:
            text: 解析するテキスト

        Returns:
            医療分析結果
        """
        doc = self.nlp(text[:50000])

        # 疾患・症状の抽出
        medical_terms = []
        disease_patterns = ['症', '病', '炎', '障害', '症候群', '疾患']

        for token in doc:
            if any(pattern in token.text for pattern in disease_patterns):
                medical_terms.append(token.text)

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
        for match in measurement_pattern.finditer(text):
            measurements.append(match.group())

        return {
            'conditions': list(set(medical_terms)),
            'treatments': list(set(treatments)),
            'measurements': list(set(measurements))
        }

    def _analyze_technical_document(self, text: str) -> Dict[str, Any]:
        """
        技術文書の特別解析

        Args:
            text: 解析するテキスト

        Returns:
            技術分析結果
        """
        doc = self.nlp(text[:50000])

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

        # 数値情報の抽出
        specifications = []
        spec_pattern = re.compile(r'\d+(\.\d+)?\s*(Hz|MHz|GHz|KB|MB|GB|TB|mm|cm|m|kg|V|W)')
        for match in spec_pattern.finditer(text):
            specifications.append(match.group())

        # 手順やステップの抽出
        steps = []
        step_patterns = ['ステップ', '手順', '手続き', '工程', '①', '②', '③', '1\\.', '2\\.', '3\\.']

        for pattern in step_patterns:
            positions = [m.start() for m in re.finditer(pattern, text)]
            for pos in positions:
                # ステップの内容を抽出
                line_end = text.find('\n', pos)
                if line_end > 0:
                    step = text[pos:line_end].strip()
                else:
                    step = text[pos:pos+100].strip()

                steps.append(step)

        return {
            'technical_terms': list(set(technical_terms))[:20],  # 最大20個
            'specifications': list(set(specifications)),
            'procedures': steps[:10]  # 最初の10ステップのみ
        }
