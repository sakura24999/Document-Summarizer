<?php

namespace App\Jobs;

use App\Models\Document;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Queue\Queueable;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;

class ProcessDocument implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    /**
     * 再試行回数
     *
     * @var int
     */
    public $tries = 3;

    /**
     * タイムアウト時間（秒）
     *
     * @var int
     */
    public $timeout = 300;

    /**
     * 処理する文書
     *
     * @var \App\Models\Document
     */
    protected $document;

    /**
     * 処理オプション
     *
     * @var array
     */
    protected $options;

    /**
     * Create a new job instance.
     */
    public function __construct(Document $document, array $options = [])
    {
        $this->document = $document;
        $this->options = $options;
    }


    /**
     * Execute the job.
     */
    public function handle(): void
    {
        try {
            Log::info("文書処理開始: ID {$this->document->id}");

            // 既存の要約がリクエストされた場合はスキップ
            if (
                !empty($this->options['reprocess']) &&
                $this->document->summaries &&
                isset($this->document->summaries[$this->options['detail_level'] ?? $this->document->detail_level])
            ) {
                Log::info("既存の要約を使用: ID {$this->document->id}");
                return;
            }

            // Python マイクロサービスに処理をリクエスト
            // 拡張して、タイムアウトと認証を追加
            $response = Http::timeout($this->timeout)
                ->withHeader('X-Token', config('services.python_api.token'))  // X-Tokenヘッダーを使用
                ->post(config('services.python_api.url') . '/api/process-document', [  // 正しいエンドポイント
                    'document_id' => $this->document->id,
                    'file_path' => $this->document->path,
                    'file_type' => pathinfo($this->document->path, PATHINFO_EXTENSION),
                    'summary_type' => $this->document->detail_level ?? 'standard',
                    'document_type' => $this->document->document_type ?? 'general',
                ]);

            if ($response->successful()) {
                Log::info("文書処理リクエスト成功: ID {$this->document->id}");

                // ステータスが "processing" のままの場合、Python サービスで非同期処理が行われる
                if ($response->json('status') === 'completed') {
                    // 同期処理が完了した場合は結果を更新
                    $this->updateDocumentWithResponse($response->json());
                }
            } else {
                Log::error("文書処理リクエスト失敗: ID {$this->document->id}, ステータスコード: {$response->status()}");
                $this->document->update([
                    'status' => 'failed',
                    'metadata' => array_merge($this->document->metadata ?? [], [
                        'error' => $response->body(),
                        'error_code' => $response->status()
                    ])
                ]);
            }
        } catch (\Exception $e) {
            Log::error("文書処理例外: ID {$this->document->id}, メッセージ: {$e->getMessage()}");
            $this->document->update([
                'status' => 'failed',
                'metadata' => array_merge($this->document->metadata ?? [], [
                    'error' => $e->getMessage(),
                    'trace' => $e->getTraceAsString()
                ])
            ]);
        }
    }

    /**
     * Python サービスからのレスポンスで文書を更新
     */
    protected function updateDocumentWithResponse(array $data): void
    {
        // 必要に応じて要約データを処理
        if (isset($data['summary'])) {
            $detailLevel = $data['detail_level'] ?? $this->document->detail_level;

            // 現在の要約を更新
            if ($detailLevel === $this->document->detail_level) {
                $this->document->summary = $data['summary'];
            }

            // 要約の詳細レベルを保存
            $summaries = $this->document->summaries ?? [];
            $summaries[$detailLevel] = $data['summary'];

            $this->document->update([
                'status' => 'completed',
                'summary' => $this->document->summary ?? $data['summary'],
                'keywords' => $data['keywords'] ?? $this->document->keywords,
                'summaries' => $summaries,
                'metadata' => array_merge($this->document->metadata ?? [], [
                    'processing_info' => $data['processing_info'] ?? null,
                ])
            ]);

            // DocumentSummary モデルにも保存（オプション）
            $this->document->documentSummaries()->updateOrCreate(
                ['detail_level' => $detailLevel],
                ['summary' => $data['summary']]
            );
        }
    }
}
