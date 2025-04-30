import re
import unicodedata
import nltk
from typing import List, Dict, Any, Optional
import spacy
from spacy.language import Language
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPPreprocessor:
    """
    NLPライブラリを使用したテキスト前処理クラス
    """

    def __init__(self, model_name: str = "ja_core_news_lg"):
        """
        初期化

        Args:
            model_name: 使用するspaCyモデル名
        """
        # spaCyモデルの読み込み
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"spaCyモデル {model_name} を読み込みました")
        except OSError:
            logger.info(f"spaCyモデル {model_name} をダウンロードしています...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        # NLTKの必要なリソースを確認
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """NLTKの必要なリソースが利用可能か確認"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def preprocess(self, text: str, normalize: bool = True,
                  remove_stopwords: bool = False) -> str:
        """
        テキストの前処理を行う

        Args:
            text: 前処理するテキスト
            normalize: Unicode正規化を行うかどうか
            remove_stopwords: ストップワードを除去するかどうか

        Returns:
            前処理されたテキスト
        """
        if not text:
            return ""

        # Unicode正規化
        if normalize:
            text = unicodedata.normalize('NFKC', text)

        # 空白の正規化
        text = self._normalize_whitespace(text)

        # 不要な文字の除去
        text = self._remove_noise(text)

        # spaCyによる処理
        doc = self.nlp(text)

        # ストップワード除去（オプション）
        if remove_stopwords:
            tokens = [token.text for token in doc if not token.is_stop]
            text = " ".join(tokens)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """空白文字の正規化"""
        # 全角スペースを半角に
        text = text.replace('\u3000', ' ')

        # 連続する空白を単一の空白に
        text = re.sub(r'\s+', ' ', text)

        # 改行の正規化
        text = text.replace('\r\n', '\n')

        return text

    def _remove_noise(self, text: str) -> str:
        """ノイズとなる文字の除去"""
        # 制御文字の除去（改行を除く）
        text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # URLの除去（オプション）
        # text = re.sub(r'https?://\S+', '', text)

        return text

    def segment_document(self, text: str) -> Dict[str, Any]:
        """
        文書をセクションや段落に分割

        Args:
            text: 分割する文書テキスト

        Returns:
            セクション構造を含む辞書
        """
        # 段落に分割
        paragraphs = [p for p in text.split('\n\n') if p.strip()]

        # 見出しの可能性のあるパターン
        heading_patterns = [
            r'^#+\s+(.+)$',                      # Markdownスタイル
            r'^第[一二三四五六七八九十]+章\s+(.+)$',  # 日本語の章番号
            r'^第\d+章\s+(.+)$',                  # 数字による章番号
            r'^(\d+\.\s+.+)$',                   # 番号付き見出し
        ]

        # セクション構造の抽出
        sections = []
        current_section = {"title": "はじめに", "content": "", "subsections": []}

        for paragraph in paragraphs:
            is_heading = False

            # 見出しかどうかをチェック
            for pattern in heading_patterns:
                match = re.match(pattern, paragraph.strip())
                if match:
                    # 現在のセクションを保存
                    if current_section["content"] or current_section["subsections"]:
                        sections.append(current_section)

                    # 新しいセクションを開始
                    current_section = {
                        "title": match.group(1) if match.groups() else paragraph.strip(),
                        "content": "",
                        "subsections": []
                    }
                    is_heading = True
                    break

            if not is_heading:
                # 既存のセクションの内容として追加
                if current_section["content"]:
                    current_section["content"] += "\n\n" + paragraph
                else:
                    current_section["content"] = paragraph

        # 最後のセクションを追加
        if current_section["content"] or current_section["subsections"]:
            sections.append(current_section)

        # 段落の解析
        doc = self.nlp(text[:100000])  # 長すぎるテキストは制限

        sentences = []
        for sent in doc.sents:
            sentences.append({
                "text": sent.text,
                "entities": [{"text": ent.text, "label": ent.label_} for ent in sent.ents]
            })

        return {
            "sections": sections,
            "paragraphs": paragraphs,
            "sentences": sentences[:1000]  # 長すぎる場合は制限
        }

    def extract_terminology(self, text: str, domain: str = None) -> List[Dict[str, Any]]:
        """
        専門用語の抽出

        Args:
            text: 用語を抽出するテキスト
            domain: 専門分野（legal, medical, technical, generalなど）

        Returns:
            抽出された専門用語のリスト
        """
        doc = self.nlp(text[:100000])  # 長すぎるテキストは制限

        # 専門分野別の重要語句パターン
        domain_patterns = {
            "legal": [r'条項', r'契約', r'当事者', r'義務', r'権利', r'法律', r'規定'],
            "medical": [r'症状', r'診断', r'治療', r'投薬', r'手術', r'病名', r'疾患'],
            "technical": [r'技術', r'仕様', r'設計', r'機能', r'システム', r'方式', r'装置']
        }

        # 専門用語候補の抽出
        terms = []

        # 1. 名詞句の抽出
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 1 and not chunk.root.is_stop:
                # 複合名詞や名詞句を追加
                term = {
                    "text": chunk.text,
                    "type": "noun_phrase",
                    "pos": chunk.root.pos_,
                    "score": 1.0
                }
                terms.append(term)

        # 2. 単独の専門用語の抽出
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 1:
                # 既に名詞句として追加済みでないかチェック
                if not any(t["text"] == token.text for t in terms):
                    term = {
                        "text": token.text,
                        "type": "single_term",
                        "pos": token.pos_,
                        "score": 0.8
                    }
                    terms.append(term)

        # 3. 専門分野に関連する用語のスコア調整
        if domain and domain in domain_patterns:
            patterns = domain_patterns[domain]
            for term in terms:
                if any(re.search(pattern, term["text"]) for pattern in patterns):
                    term["score"] *= 1.5  # スコアを50%上げる
                    term["domain_specific"] = True

        # スコア順にソート
        terms.sort(key=lambda x: x["score"], reverse=True)

        return terms[:100]  # 上位100件を返す

    def create_custom_tokenizer(self):
        """
        カスタムトークナイザーコンポーネントの作成

        Returns:
            カスタムトークナイザー関数
        """
        # 日本語の分かち書きをカスタマイズするコンポーネント
        @Language.component("custom_japanese_tokenizer")
        def custom_tokenizer(doc):
            # ここにカスタム処理を実装
            return doc

        # コンポーネントをNLPパイプラインに追加
        self.nlp.add_pipe("custom_japanese_tokenizer", before="parser")

        return self.nlp
