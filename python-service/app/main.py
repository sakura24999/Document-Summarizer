from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import os
import uuid
import logging
import httpx
from datetime import datetime

# プロセッサーモジュール
from app.processors.base import BaseDocumentProcessor
from app.processors.pdf_processor import PDFProcessor
from app.processors.docx_processor import DOCXProcessor
from app.processors.text_processor import TEXTProcessor

# 要約モジュール
#from app.summarizers.claude_summarizer import ClaudeSummarizer

from app.summarizers.openai_summarizer import OpenAISummarizer

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/app/logs/python_api.log',  # コンテナ内のログ保存先
    filemode='a'
)

logger = logging.getLogger(__name__)

# アプリケーションの初期化
app = FastAPI(title="Document Summarizer API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 認証トークン
API_TOKEN = os.getenv("API_TOKEN", "your-secret-token")

# タスク状態を保持する辞書
tasks = {}

# リクエストモデル
class DocumentRequest(BaseModel):
    document_id: int
    file_path: str
    file_type: str
    summary_type: str = "standard"  # 'brief', 'standard', 'detailed'
    document_type: str = "general"  # 'legal', 'technical', 'medical', 'academic', 'business', 'general'

# レスポンスモデル
class TaskResponse(BaseModel):
    task_id: str
    status: str = "pending"

class SummaryResult(BaseModel):
    summary: str
    keywords: List[str]
    metadata: Dict[str, Any]

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[SummaryResult] = None
    error: Optional[str] = None

# 認証依存関数
async def verify_token(x_token: str = Header(...)):
    if x_token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
        )
    return x_token

# プロセッサーのファクトリー関数
def get_processor(file_type: str) -> BaseDocumentProcessor:
    file_type = file_type.lower()
    if file_type == 'pdf':
        return PDFProcessor()
    elif file_type in ['docx', 'doc']:
        return DOCXProcessor()
    elif file_type in ['txt', 'text']:
        return TEXTProcessor()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# バックグラウンドタスク
async def process_document_task(task_id: str, request: DocumentRequest):
    try:
        tasks[task_id]["status"] = "processing"

        # ファイルプロセッサーの取得
        processor = get_processor(request.file_type)

        # ファイルパスの確認
        file_path = request.file_path
        if not os.path.exists(file_path):
            # Laravelのストレージパスの場合は変換
            storage_path = os.getenv("LARAVEL_STORAGE_PATH", "/var/www/html/storage/app")
            file_path = os.path.join(storage_path, request.file_path)

        # 文書の処理
        extracted_text, metadata = processor.process(file_path)

        # 要約クラスの初期化
        #summarizer = ClaudeSummarizer()
        summarizer = OpenAISummarizer()

        # 要約と結果の取得
        summary_text, keywords = summarizer.summarize(
            text=extracted_text,
            document_type=request.document_type,
            summary_type=request.summary_type,
        )

        # タスク結果の更新
        tasks[task_id].update({
            "status": "completed",
            "result": {
                "summary": summary_text,
                "keywords": keywords,
                "metadata": metadata,
            },
            "completed_at": datetime.now().isoformat(),
        })

        # Laravel側のドキュメントステータスも更新
        await update_document_status(
            request.document_id,
            "completed",
            summary=summary_text,
            keywords=keywords
        )

        logger.info(f"Document {request.document_id} processing completed: {task_id}")

    except Exception as e:
        logger.error(f"Error processing document {request.document_id}: {str(e)}")
        tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        })

        # エラー時にもLaravel側に通知
        await update_document_status(
            request.document_id,
            "error",
            message=f"処理中にエラーが発生しました: {str(e)}"
        )
# エンドポイント定義
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/process-document", response_model=TaskResponse, dependencies=[Depends(verify_token)])
async def process_document(request: DocumentRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    # タスク情報の初期化
    tasks[task_id] = {
        "document_id": request.document_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }

    # バックグラウンドタスクの追加
    background_tasks.add_task(process_document_task, task_id, request)

    logger.info(f"Document {request.document_id} processing started: {task_id}")

    return {"task_id": task_id, "status": "pending"}

@app.get("/api/task-status/{task_id}", response_model=TaskStatusResponse, dependencies=[Depends(verify_token)])
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task_info = tasks[task_id]

    response = {
        "task_id": task_id,
        "status": task_info["status"],
    }

    if "result" in task_info:
        response["result"] = task_info["result"]

    if "error" in task_info:
        response["error"] = task_info["error"]

    return response

# メインエントリーポイント
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# Laravel APIと通信するためのヘルパー関数
async def update_document_status(document_id, status, message=None, summary=None, keywords=None):
    """ドキュメントのステータスをLaravel側で更新する"""
    try:
        async with httpx.AsyncClient() as client:
            data = {
                "status": status,
            }

            if message:
                data["message"] = message

            if summary:
                data["summary"] = summary

            if keywords:
                data["keywords"] = keywords

            # Laravelサーバーのエンドポイント
            url = f"http://web/api/documents/{document_id}/update-status"
            response = await client.post(url, json=data, timeout=10.0)

            if response.status_code != 200:
                logger.error(f"Failed to update document status: {response.text}")

            return response.status_code == 200
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}")
        return False
