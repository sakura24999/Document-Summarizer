from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import requests
import os
import logging
from app.processors import get_processor_for_file
from app.summarizers.claude_summarizer import ClaudeSummarizer

app = FastAPI(title="文書要約マイクロサービス")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数からLaravel APIのURLとトークンを取得
LARAVEL_API_URL = os.environ.get("LARAVEL_API_URL", "http://localhost:8000/api")
API_TOKEN = os.environ.get("API_TOKEN", "")

# Anthropic API Keyの設定
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# サマライザーの初期化
summarizer = ClaudeSummarizer(api_key=ANTHROPIC_API_KEY)

class DocumentRequest(BaseModel):
    document_id: int
    file_path: str
    options: Optional[Dict[str, Any]] = {}

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy"}

@app.post("/process")
async def process_document(request: DocumentRequest, background_tasks: BackgroundTasks):
    """文書処理を非同期で開始"""
    # バックグラウンドで処理を実行
    background_tasks.add_task(process_document_task, request)
    return {"status": "processing", "document_id": request.document_id}

async def process_document_task(request: DocumentRequest):
    """文書処理のメインタスク"""
    document_id = request.document_id
    file_path = request.file_path
    options = request.options

    try:
        # ファイルタイプに基づいたプロセッサの取得
        processor = get_processor_for_file(file_path)
        if not processor:
            raise ValueError(f"Unsupported file type: {file_path}")

        # テキスト抽出
        logger.info(f"Extracting text from {file_path}")
        text, metadata = processor.extract_text(file_path)

        # 文書タイプの自動判定（オプションで指定されていない場合）
        document_type = options.get('document_type', 'auto')
        if document_type == 'auto':
            document_type = detect_document_type(text)

        # 要約の詳細レベル
        detail_level = options.get('detail_level', 'standard')

        # 要約生成
        logger.info(f"Generating summary for document {document_id} with level {detail_level}")
        summary_result = summarizer.summarize(
            text=text,
            document_type=document_type,
            detail_level=detail_level,
            extract_keywords=options.get('extract_keywords', True)
        )

        # 結果をLaravelに送信
        send_result_to_laravel(document_id, summary_result, metadata, document_type)

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        # エラー情報をLaravelに送信
        send_error_to_laravel(document_id, str(e))

def detect_document_type(text):
    """文書タイプの自動判定"""
    # 実際の実装では、テキスト分析によって文書タイプを判定
    # 簡易的な実装例：
    keywords = {
        'legal': ['法律', '契約', '条項', '規約', '法令', '法的'],
        'technical': ['技術', '仕様', 'システム', '実装', 'アーキテクチャ'],
        'medical': ['医療', '診断', '治療', '患者', '病院', '臨床'],
        'academic': ['研究', '論文', '調査', '分析', '理論', '実験'],
        'business': ['ビジネス', '戦略', '市場', '売上', '利益', '顧客']
    }

    counts = {doc_type: 0 for doc_type in keywords}
    for doc_type, terms in keywords.items():
        for term in terms:
            counts[doc_type] += text.count(term)

    # 最も一致数の多いタイプを返す
    max_type = max(counts.items(), key=lambda x: x[1])
    if max_type[1] > 0:
        return max_type[0]

    # デフォルト値
    return 'general'

def send_result_to_laravel(document_id, summary_result, metadata, document_type):
    """要約結果をLaravelに送信"""
    url = f"{LARAVEL_API_URL}/documents/{document_id}/summary"

    data = {
        "status": "completed",
        "summary": summary_result.get("summary", ""),
        "keywords": summary_result.get("keywords", []),
        "summaries": {
            summary_result.get("detail_level", "standard"): summary_result.get("summary", "")
        },
        "metadata": {
            **metadata,
            "document_type": document_type,
            "ai_processing_info": summary_result.get("processing_info", {})
        }
    }

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully sent summary for document {document_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send summary to Laravel: {str(e)}")

def send_error_to_laravel(document_id, error_message):
    """エラー情報をLaravelに送信"""
    url = f"{LARAVEL_API_URL}/documents/{document_id}/summary"

    data = {
        "status": "failed",
        "error": error_message
    }

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully sent error for document {document_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send error to Laravel: {str(e)}")
