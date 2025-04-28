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
from app.summarizers.claude_summarizer import ClaudeSummarizer

from app.summarizers.specialized_summarizer import SpecializedSummarizer

#from app.summarizers.openai_summarizer import OpenAISummarizer

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

        # デバッグ用に情報を出力
        print(f"Received request: {request.dict()}")

        # ファイルプロセッサーの取得
        processor = get_processor(request.file_type)

        # ファイルパスの確認
        file_path = request.file_path
        print(f"Original file_path: {file_path}")

        # もしファイルパスがdocumentsで始まっている場合は、/var/www/html/storage/app/を先頭に追加
        if file_path.startswith('documents/'):
            file_path = f"/var/www/html/storage/app/app/private/{file_path}"
            print(f"Modified file_path: {file_path}")

        # ファイルが存在するか確認
        import os
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            # 別のパスも試してみる
            alternate_paths = [
                f"/var/www/html/storage/app/app/private/documents/{os.path.basename(file_path)}",
                f"/app/storage/app/app/private/{file_path}",
                f"/var/www/html/storage/app/{file_path}",
                f"/app/{file_path}"
            ]

            for alt_path in alternate_paths:
                print(f"Trying alternate path: {alt_path}")
                if os.path.exists(alt_path):
                    file_path = alt_path
                    print(f"File found at: {file_path}")
                    break
            else:
                raise FileNotFoundError(f"File not found in any of the expected locations: {file_path}")

        print(f"Processing file: {file_path}")

        # 文書の処理
        result = processor.process(file_path)
        extracted_text = result["text"]
        metadata = result["metadata"]
        print(f"Extracted text length: {len(extracted_text)} characters")

        # 要約クラスの初期化
        # summarizer = OpenAISummarizer()
        summarizer = ClaudeSummarizer()
        logger.info("サマライザー初期化成功")
        # 要約と結果の取得
        summary_result = summarizer.summarize(
            text=extracted_text,
            document_type=request.document_type,
            detail_level=request.summary_type,  # summary_type から detail_level に変更
            extract_keywords=True
        )

        # 結果から要約テキストとキーワードを取得
        summary_text = summary_result.get("summary", "")
        keywords = summary_result.get("keywords", [])
        logger.info("要約生成完了")

        logger.info("要約結果の詳細:")
        logger.info(f"要約テキスト長: {len(summary_text)} 文字")
        logger.info(f"キーワード: {keywords}")

        print(f"Summary generated: {len(summary_text)} characters")
        print(f"Keywords: {keywords}")

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

        # URLリスト
        logger.info("ステータス更新APIリクエスト準備開始")
        urls_to_try = [
            f"http://web/api/documents/{request.document_id}/update-status",
            f"http://document-summarizer-web/api/documents/{request.document_id}/update-status",
            f"http://nginx/api/documents/{request.document_id}/update-status",
            f"http://web:80/api/documents/{request.document_id}/update-status"
        ]

        data = {
            "status": "completed",
            "summary": summary_text,
            "keywords": keywords
        }

        logger.info(f"更新データ: status=completed, summary_length={len(summary_text)}, keywords_count={len(keywords)}")

        success = False
        last_error = None
        logger.info("API通信開始")

        for url in urls_to_try:
            try:
                logger.info(f"URL試行: {url}")

                async with httpx.AsyncClient() as client:
                    logger.info("HTTPリクエスト送信中...")
                    response = await client.post(
                        url,
                        json=data,
                        timeout=20.0,
                        headers={"Accept": "application/json", "Content-Type": "application/json"}
                    )
                    logger.info(f"レスポンス受信: ステータスコード={response.status_code}")
                    logger.info(f"レスポンス本文: {response.text}")

                    if response.status_code == 200:
                        logger.info("ステータス更新成功!")
                        success = True
                        break
                    else:
                        logger.error(f"ステータス更新失敗: HTTP {response.status_code}")
            except Exception as e:
                last_error = e
                logger.error(f"通信エラー: {type(e).__name__}: {str(e)}")
                # エラーの詳細情報を出力
                import traceback
                logger.error(f"エラー詳細: {traceback.format_exc()}")
                continue

        if success:
            logger.info(f"Document {request.document_id} ステータス更新完了")
        else:
            logger.error(f"Document {request.document_id} ステータス更新: すべての接続試行が失敗")
            if last_error:
                logger.error(f"最後のエラー: {type(last_error).__name__}: {str(last_error)}")
            raise Exception(f"All API connection attempts failed: {str(last_error)}")

        logger.info(f"Document {request.document_id} processing completed: {task_id}")
        return {"status": "success", "task_id": task_id}

    except Exception as e:
        logger.error(f"要約中のエラー（詳細）: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"スタックトレース: {traceback.format_exc()}")

        print(f"Error in process_document_task: {str(e)}")
        logger.error(f"Error processing document {request.document_id}: {str(e)}")

        tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        })

        # エラー時にもLaravel側に通知
        try:
            # URLリスト
            urls_to_try = [
                f"http://web/api/documents/{request.document_id}/update-status",
                f"http://document-summarizer-web/api/documents/{request.document_id}/update-status",
                f"http://nginx/api/documents/{request.document_id}/update-status"
            ]

            error_data = {
                "status": "error",
                "message": f"処理中にエラーが発生しました: {str(e)}"
            }

            for url in urls_to_try:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=error_data, timeout=10.0)
                        if response.status_code == 200:
                            break
                except Exception:
                    continue
        except Exception as notify_error:
            print(f"Failed to notify error: {str(notify_error)}")

        raise Exception(f"文書の要約中にエラーが発生しました: {str(e)}")

        return {"status": "error", "error": str(e), "task_id": task_id}

# 専門文書処理タスク
async def process_specialized_document_task(task_id: str, request: SpecializedDocumentRequest):
    try:
        tasks[task_id]["status"] = "processing"

        # デバッグ用に情報を出力
        print(f"Received specialized request: {request.dict()}")

        # ファイルプロセッサーの取得
        processor = get_processor(request.file_type)

        # ファイルパスの確認
        file_path = request.file_path
        print(f"Original file_path: {file_path}")

        # もしファイルパスがdocumentsで始まっている場合は、/var/www/html/storage/app/を先頭に追加
        if file_path.startswith('documents/'):
            file_path = f"/var/www/html/storage/app/app/private/{file_path}"
            print(f"Modified file_path: {file_path}")

        # ファイルが存在するか確認
        import os
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            # 別のパスも試してみる
            alternate_paths = [
                f"/var/www/html/storage/app/app/private/documents/{os.path.basename(file_path)}",
                f"/app/storage/app/app/private/{file_path}",
                f"/var/www/html/storage/app/{file_path}",
                f"/app/{file_path}"
            ]

            for alt_path in alternate_paths:
                print(f"Trying alternate path: {alt_path}")
                if os.path.exists(alt_path):
                    file_path = alt_path
                    print(f"File found at: {file_path}")
                    break
            else:
                raise FileNotFoundError(f"File not found in any of the expected locations: {file_path}")

        print(f"Processing file: {file_path}")

        # 文書の処理
        result = processor.process(file_path)
        extracted_text = result["text"]
        metadata = result["metadata"]
        print(f"Extracted text length: {len(extracted_text)} characters")

        # 専門的な要約の生成
        logger.info("専門的要約生成を開始")
        summary_result = specialized_summarizer.summarize(
            text=extracted_text,
            document_type=request.document_type,
            document_subtype=request.document_subtype,
            detail_level=request.summary_type,
            extract_keywords=True
        )

        # 結果から要約テキストとキーワードを取得
        summary_text = summary_result.get("summary", "")
        keywords = summary_result.get("keywords", [])

        # 専門分析結果があれば取得
        specialized_analysis = {}
        if request.document_type == 'legal' and 'legal_analysis' in summary_result:
            specialized_analysis = summary_result['legal_analysis']
        elif request.document_type == 'medical' and 'medical_analysis' in summary_result:
            specialized_analysis = summary_result['medical_analysis']

        logger.info("専門的要約生成完了")
        logger.info(f"要約テキスト長: {len(summary_text)} 文字")
        logger.info(f"キーワード: {keywords}")
        logger.info(f"専門分析: {specialized_analysis.keys() if specialized_analysis else 'なし'}")

        # タスク結果の更新
        tasks[task_id].update({
            "status": "completed",
            "result": {
                "summary": summary_text,
                "keywords": keywords,
                "metadata": metadata,
                "specialized_analysis": specialized_analysis
            },
            "completed_at": datetime.now().isoformat(),
        })

        # URLリスト
        logger.info("ステータス更新APIリクエスト準備開始")
        urls_to_try = [
            f"http://web/api/documents/{request.document_id}/update-status",
            f"http://document-summarizer-web/api/documents/{request.document_id}/update-status",
            f"http://nginx/api/documents/{request.document_id}/update-status",
            f"http://web:80/api/documents/{request.document_id}/update-status"
        ]

        data = {
            "status": "completed",
            "summary": summary_text,
            "keywords": keywords,
            "specialized_analysis": specialized_analysis
        }

        logger.info(f"更新データ: status=completed, summary_length={len(summary_text)}, keywords_count={len(keywords)}")

        success = False
        last_error = None
        logger.info("API通信開始")

        for url in urls_to_try:
            try:
                logger.info(f"URL試行: {url}")

                async with httpx.AsyncClient() as client:
                    logger.info("HTTPリクエスト送信中...")
                    response = await client.post(
                        url,
                        json=data,
                        timeout=20.0,
                        headers={"Accept": "application/json", "Content-Type": "application/json"}
                    )
                    logger.info(f"レスポンス受信: ステータスコード={response.status_code}")
                    logger.info(f"レスポンス本文: {response.text}")

                    if response.status_code == 200:
                        logger.info("ステータス更新成功!")
                        success = True
                        break
                    else:
                        logger.error(f"ステータス更新失敗: HTTP {response.status_code}")
            except Exception as e:
                last_error = e
                logger.error(f"通信エラー: {type(e).__name__}: {str(e)}")
                # エラーの詳細情報を出力
                import traceback
                logger.error(f"エラー詳細: {traceback.format_exc()}")
                continue

        if success:
            logger.info(f"Document {request.document_id} ステータス更新完了")
        else:
            logger.error(f"Document {request.document_id} ステータス更新: すべての接続試行が失敗")
            if last_error:
                logger.error(f"最後のエラー: {type(last_error).__name__}: {str(last_error)}")
            raise Exception(f"All API connection attempts failed: {str(last_error)}")

        logger.info(f"Specialized document {request.document_id} processing completed: {task_id}")
        return {"status": "success", "task_id": task_id}

    except Exception as e:
        logger.error(f"専門要約中のエラー（詳細）: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"スタックトレース: {traceback.format_exc()}")

        print(f"Error in process_specialized_document_task: {str(e)}")
        logger.error(f"Error processing specialized document {request.document_id}: {str(e)}")

        tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        })

        # エラー時にもLaravel側に通知
        try:
            # URLリスト
            urls_to_try = [
                f"http://web/api/documents/{request.document_id}/update-status",
                f"http://document-summarizer-web/api/documents/{request.document_id}/update-status",
                f"http://nginx/api/documents/{request.document_id}/update-status"
            ]

            error_data = {
                "status": "error",
                "message": f"専門文書処理中にエラーが発生しました: {str(e)}"
            }

            for url in urls_to_try:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=error_data, timeout=10.0)
                        if response.status_code == 200:
                            break
                except Exception:
                    continue
        except Exception as notify_error:
            print(f"Failed to notify error: {str(notify_error)}")

        raise Exception(f"専門文書の要約中にエラーが発生しました: {str(e)}")

# エンドポイント定義
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/api/process-document", response_model=TaskResponse, dependencies=[Depends(verify_token)])
async def process_document(request: DocumentRequest, background_tasks: BackgroundTasks):
    print(f"Received request: {request.dict()}")
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
        logger.info(f"ドキュメント{document_id}のステータス更新開始: {status}")

        async with httpx.AsyncClient() as client:
            data = {
                "status": status,
            }

            if message:
                data["message"] = message

            if summary:
                data["summary"] = summary
                logger.info(f"要約文字数: {len(summary)}")

            if keywords:
                data["keywords"] = keywords
                logger.info(f"キーワード数: {len(keywords)}")

            logger.info(f"更新データ準備完了: {data.keys()}")
            # Docker Composeのサービス名で複数の可能性を試す
            urls_to_try = [
                f"http://web/api/documents/{document_id}/update-status",
                f"http://document-summarizer-web/api/documents/{document_id}/update-status",
                f"http://nginx/api/documents/{document_id}/update-status",
                f"http://localhost:8080/api/documents/{document_id}/update-status"
            ]

            success = False
            for url in urls_to_try:
                try:
                    logger.info(f"URL試行: {url}")
                    response = await client.post(url, json=data, timeout=30.0)
                    logger.info(f"レスポンス: ステータスコード={response.status_code}")
                    logger.info(f"レスポンス本文: {response.text[:200]}...")

                    if response.status_code == 200:
                        logger.info(f"ドキュメント{document_id}のステータス更新成功")
                        success = True
                        break
                    else:
                        logger.error(f"ステータス更新API失敗: HTTP {response.status_code}")
                except Exception as e:
                    logger.error(f"URL {url} との通信エラー: {type(e).__name__}: {str(e)}")
                    # スタックトレースも出力
                    import traceback
                    logger.error(f"スタックトレース: {traceback.format_exc()}")
                    continue
            if success:
                logger.info(f"ドキュメント{document_id}の最終ステータス更新完了")
            else:
                logger.error(f"ドキュメント{document_id}の更新失敗: すべてのURLで接続エラー")

            return success
    except Exception as e:
        logger.error(f"ステータス更新処理全体でのエラー: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"スタックトレース: {traceback.format_exc()}")
        return False
