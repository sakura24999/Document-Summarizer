# app/nlp_preprocessor.py
# これは単にapp.summarizers.nlp_preprocessorからクラスをリエクスポートするブリッジファイルです

from app.summarizers.nlp_preprocessor import NLPPreprocessor

# クラスをそのまま再エクスポート
# これにより、app.main.pyからインポートする際に正しく解決されます
