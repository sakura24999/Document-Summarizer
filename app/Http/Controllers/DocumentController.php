<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Document;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Storage;
use App\Jobs\ProcessDocument;
use App\Http\Requests\StoreDocumentRequest;
use Illuminate\Support\Facades\Log;

class DocumentController extends Controller
{
    public function __construct()
    {

    }
    /**
     * 文書一覧の表示
     */
    public function index()
    {
        $documents = Auth::user()->documents()
            ->latest()
            ->paginate(10);

        return view('documents.index', compact('documents'));
    }

    /**
     * 新規文書アップロードフォームを表示
     */
    public function create()
    {
        return view('documents.create');
    }

    protected function authorize($ability, $arguments = [])
    {
        if (Auth::user()->cannot($ability, $arguments)) {
            abort(403, 'Unauthorized action.');
        }
    }

    /**
     * 文書のアップロードと処理
     */
    public function store(StoreDocumentRequest $request)
    {
        // ファイルの保存
        $file = $request->file('document');
        $path = $file->store('documents');

        // ファイル内容の読み取り（テキスト抽出はPythonサービスで行うため省略可）
        $content = ''; // 実際のプロジェクトでは初期値として設定

        // 文書情報をDBに保存
        $document = Document::create([
            'user_id' => Auth::id(),
            'filename' => $file->getClientOriginalName(),
            'path' => $path,
            'content' => $content,
            'status' => 'processing',
            'detail_level' => $request->input('detail_level', 'standard'),
            'document_type' => $request->input('summary_type', 'auto'),
            'options' => [
                'extract_keywords' => $request->has('extract_keywords'),
                'document_type' => $request->input('summary_type', 'auto'),
            ]
        ]);

        // 非同期ジョブの発行
        ProcessDocument::dispatch($document);

        if ($request->ajax() || $request->wantsJson()) {
            return response()->json([
                'status' => 'success',
                'document_id' => $document->id
            ]);
        }

        return redirect()->route('documents.show', $document)
            ->with('status', '文書を処理中です。しばらくお待ちください。');
    }

    /**
     * 文書の詳細と要約の表示
     */
    public function show($id)
    {
        $document = Document::findOrFail($id);
        // 権限チェック
        $this->authorize('view', $document);

        return view('documents.show', compact('document'));
    }

    /**
     * 要約の詳細レベル変更
     */
    public function updateDetailLevel(Request $request, Document $document)
    {
        // 権限チェック
        $this->authorize('update', $document);

        $level = $request->input('level');

        // 対応するレベルの要約がすでに生成されているかチェック
        if (isset($document->summaries[$level])) {
            // 既存の要約を取得
            $summary = $document->summaries[$level];
        } else {
            // 新しい詳細レベルで要約を生成するジョブを発行
            ProcessDocument::dispatch($document, ['detail_level' => $level, 'reprocess' => true]);

            return response()->json([
                'status' => 'processing',
                'message' => '新しい詳細レベルで要約を生成中です。'
            ]);
        }

        // 現在の詳細レベルを更新
        $document->detail_level = $level;
        $document->save();

        return response()->json([
            'status' => 'success',
            'summary' => $summary
        ]);
    }

    /**
     * 文書の削除
     */
    public function destroy(Document $document)
    {
        // 権限チェック
        $this->authorize('delete', $document);

        // ストレージからファイルを削除
        if (Storage::exists($document->path)) {
            Storage::delete($document->path);
        }

        // DBから削除
        $document->delete();

        return redirect()->route('documents.index')
            ->with('status', '文書を削除しました。');
    }

    public function updateStatus(Request $request, Document $document)
    {
        $validated = $request->validate([
            'status' => 'required|string|in:processing,completed,error',
            'message' => 'nullable|string',
            'summary' => 'nullable|string',
            'keywords' => 'nullable|array'
        ]);

        $document->status = $validated['status'];

        if (isset($validated['summary'])) {
            $document->summary = $validated['summary'];
        }

        if (isset($validated['keywords'])) {
            $document->keywords = $validated['keywords'];
        }

        // エラーメッセージがあればメタデータに保存
        if (isset($validated['message'])) {
            $metadata = json_decode($document->metadata ?? '{}', true) ?: [];
            $metadata['error_message'] = $validated['message'];
            $document->metadata = json_encode($metadata);
        }

        $document->save();

        return response()->json(['success' => true]);
    }

    public function retryProcessing(Document $document)
    {
        try {
            // ドキュメントのステータスをprocessingに戻す
            $document->status = 'processing';
            $document->save();

            // ProcessDocumentジョブを再ディスパッチ
            ProcessDocument::dispatch($document);

            return response()->json(['status' => 'success']);
        } catch (\Exception $e) {
            Log::error('Retry processing failed: ' . $e->getMessage());
            return response()->json(['status' => 'error', 'message' => $e->getMessage()], 500);
        }
    }

    public function summarize(Document $document)
    {
        // 要約処理のロジックをここに実装
        // 例えば：
        try {
            $document->status = 'processing';
            $document->save();
            ProcessDocument::dispatch($document);
            return view('documents.summary', compact('document'));
        } catch (\Exception $e) {
            Log::error('Summary processing failed: ' . $e->getMessage());
            return back()->with('error', 'ドキュメントの要約処理に失敗しました。');
        }
    }
}
