import re
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from transformers import AutoTokenizer, AutoModel
import torch

class DocumentClassifier:
    """
    SpacyとTransformersを活用した高度な文書分類クラス
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        分類器の初期化

        Args:
            model_path: 事前学習済みモデルのパス（Noneの場合はルールベース分類を使用）
        """
        # 法務文書の特徴パターン
        self.legal_patterns = {
            'contract': [r'当事者', r'甲.*乙', r'契約書', r'合意する', r'本契約', r'以下の条件'],
            'case_law': [r'判決', r'判示', r'裁判所', r'原告', r'被告', r'控訴', r'上告'],
            'legislation': [r'第\d+条', r'法律', r'規則', r'施行', r'政令', r'条例']
        }

        # 医療文書の特徴パターン
        self.medical_patterns = {
            'diagnosis': [r'診断', r'症状', r'処方', r'投与', r'mg', r'検査', r'陽性', r'陰性'],
            'clinical_note': [r'既往歴', r'主訴', r'現病歴', r'所見', r'診察'],
            'research_paper': [r'臨床試験', r'有意差', r'p値', r'被験者', r'対照群', r'有効性']
        }

        # spaCyモデルの読み込み
        try:
            # 日本語モデルをロード
            self.nlp = spacy.load("ja_core_news_lg")
            print("日本語spaCyモデルを読み込みました")
        except OSError:
            # モデルがインストールされていない場合は、ダウンロードしてからロード
            print("日本語spaCyモデルをダウンロードしています...")
            spacy.cli.download("ja_core_news_lg")
            self.nlp = spacy.load("ja_core_news_lg")

        # BERT系の文埋め込みモデルの読み込み
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
        self.bert_model = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-v3")

        # 機械学習モデルの読み込み（存在すれば）
        self.ml_classifier = None
        self.vectorizer = None

        if model_path and os.path.exists(model_path):
            try:
                model_data = pickle.load(open(model_path, 'rb'))
                self.ml_classifier = model_data['classifier']
                self.vectorizer = model_data['vectorizer']
                print(f"機械学習モデルを {model_path} から読み込みました")
            except Exception as e:
                print(f"モデル読み込みエラー: {e}")

    def classify(self, text: str) -> Dict[str, Any]:
        """
        文書テキストから専門分野と文書タイプを推定

        Args:
            text: 分析する文書テキスト

        Returns:
            専門分野と文書タイプを含む辞書
        """
        # spaCyによるテキスト解析
        sample_text = text[:25000]  # 分析のために先頭部分だけを使用
        doc = self.nlp(sample_text)

        # 機械学習モデルが利用可能な場合はそれを使用
        if self.ml_classifier and self.vectorizer:
            domain, subtype, confidence = self._predict_with_ml(text)
        else:
            # ルールベースの分類を実行
            domain, subtype, confidence = self._predict_with_rules(text, doc)

        # 固有表現の抽出
        entities = self._extract_entities(doc)

        # キーワードの抽出
        keywords = self._extract_keywords(doc)

        return {
            'domain': domain,
            'subtype': subtype,
            'confidence': confidence,
            'entities': entities,
            'keywords': keywords
        }

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        BERTを使用してテキストの埋め込みを取得

        Args:
            text: 埋め込みを取得するテキスト

        Returns:
            テキストの埋め込みベクトル
        """
        # テキストをトークン化
        inputs = self.tokenizer(text[:512], return_tensors="pt",
                               padding=True, truncation=True, max_length=512)

        # 埋め込みを計算
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # [CLS]トークンの最終隠れ状態を取得
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings[0]

    def _predict_with_ml(self, text: str) -> Tuple[str, str, float]:
        """
        機械学習モデルを使用して文書の分類を予測

        Args:
            text: 分類するテキスト

        Returns:
            (ドメイン, サブタイプ, 信頼度)のタプル
        """
        # テキスト特徴量の抽出
        text_features = self.vectorizer.transform([text[:10000]])

        # ドメインの予測
        domain_pred = self.ml_classifier.predict(text_features)[0]
        confidence = max(self.ml_classifier.predict_proba(text_features)[0])

        # サブタイプの予測（ドメインごとに専用の分類器を持つ理想的な場合）
        # 簡略化のため、ここではルールベースでサブタイプを決定
        if domain_pred == 'legal':
            subtype = self._determine_legal_subtype(text)
        elif domain_pred == 'medical':
            subtype = self._determine_medical_subtype(text)
        else:
            subtype = None

        return domain_pred, subtype, confidence

    def _predict_with_rules(self, text: str, doc) -> Tuple[str, str, float]:
        """
        ルールベースで文書の分類を予測

        Args:
            text: 分類するテキスト
            doc: spaCyのドキュメントオブジェクト

        Returns:
            (ドメイン, サブタイプ, 信頼度)のタプル
        """
        # 各専門分野・文書タイプの一致スコアを計算
        legal_scores = self._calculate_domain_scores(text, self.legal_patterns)
        medical_scores = self._calculate_domain_scores(text, self.medical_patterns)

        # 最高スコアの専門分野とサブタイプを決定
        domain_scores = {
            'legal': max(legal_scores.values()) if legal_scores else 0,
            'medical': max(medical_scores.values()) if medical_scores else 0,
            # その他の専門分野...
        }

        # 最も高いスコアの専門分野を選択
        domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[domain]

        # NLPベースの特徴も考慮
        # 例: 法律関連の単語の出現頻度
        legal_tokens = [token.text for token in doc if token.text.lower() in
                        ['法律', '契約', '条項', '規定', '義務', '権利', '法的']]

        # 例: 医療関連の単語の出現頻度
        medical_tokens = [token.text for token in doc if token.text.lower() in
                         ['診断', '症状', '治療', '患者', '医療', '病院', '処方']]

        # トークン出現頻度に基づく信頼度スコアの調整
        if domain == 'legal' and len(legal_tokens) > 0:
            confidence = max(confidence, len(legal_tokens) / len(doc) * 100)
        elif domain == 'medical' and len(medical_tokens) > 0:
            confidence = max(confidence, len(medical_tokens) / len(doc) * 100)

        # その専門分野内で最も高いスコアのサブタイプを選択
        if domain == 'legal':
            subtype = max(legal_scores, key=legal_scores.get) if legal_scores else None
        elif domain == 'medical':
            subtype = max(medical_scores, key=medical_scores.get) if medical_scores else None
        else:
            subtype = None

        return domain, subtype, confidence

    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """
        spaCyを使用して固有表現を抽出

        Args:
            doc: spaCyのドキュメントオブジェクト

        Returns:
            タイプごとの固有表現の辞書
        """
        entities = {}

        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []

            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)

        return entities

    def _extract_keywords(self, doc, max_keywords: int = 20) -> List[str]:
        """
        重要なキーワードを抽出

        Args:
            doc: spaCyのドキュメントオブジェクト
            max_keywords: 抽出するキーワードの最大数

        Returns:
            キーワードのリスト
        """
        # 名詞、固有名詞、形容詞を抽出
        keywords = []

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and len(token.text) > 1:
                keywords.append(token.text)

        # 頻度でソート（より高度な方法はTF-IDFなど）
        keyword_freq = {}
        for word in keywords:
            if word in keyword_freq:
                keyword_freq[word] += 1
            else:
                keyword_freq[word] = 1

        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [k[0] for k in sorted_keywords[:max_keywords]]

    def _calculate_domain_scores(self, text: str, patterns: Dict[str, List[str]]) -> Dict[str, float]:
        """
        各文書タイプのパターンマッチングスコアを計算

        Args:
            text: 分析する文書テキスト
            patterns: タイプ別パターンの辞書

        Returns:
            タイプごとのスコアを含む辞書
        """
        scores = {}

        for doc_type, pattern_list in patterns.items():
            score = 0
            for pattern in pattern_list:
                matches = re.findall(pattern, text)
                score += len(matches)

            if score > 0:
                # テキスト長で正規化
                scores[doc_type] = score / (len(text) / 1000)

        return scores

    def _determine_legal_subtype(self, text: str) -> str:
        """
        法務文書のサブタイプを判定

        Args:
            text: 分析するテキスト

        Returns:
            法務文書のサブタイプ
        """
        legal_scores = self._calculate_domain_scores(text, self.legal_patterns)
        return max(legal_scores, key=legal_scores.get) if legal_scores else None

    def _determine_medical_subtype(self, text: str) -> str:
        """
        医療文書のサブタイプを判定

        Args:
            text: 分析するテキスト

        Returns:
            医療文書のサブタイプ
        """
        medical_scores = self._calculate_domain_scores(text, self.medical_patterns)
        return max(medical_scores, key=medical_scores.get) if medical_scores else None

    def train(self, texts: List[str], labels: List[Tuple[str, str]], model_path: str = 'classifier_model.pkl'):
        """
        テキストデータと正解ラベルから分類器を学習

        Args:
            texts: 学習用のテキストリスト
            labels: (ドメイン, サブタイプ)のタプルからなるラベルリスト
            model_path: 学習済みモデルの保存先パス
        """
        # 特徴量抽出器の作成
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)

        # ドメインラベルの抽出
        y = [label[0] for label in labels]

        # 分類器の学習
        self.ml_classifier = RandomForestClassifier(n_estimators=100)
        self.ml_classifier.fit(X, y)

        # モデルの保存
        model_data = {
            'classifier': self.ml_classifier,
            'vectorizer': self.vectorizer
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"モデルを {model_path} に保存しました")
