@extends('layouts.app')

@section('title', $document->filename)

@section('styles')
    @vite('resources/css/css/app.scss')
@endsection

@section('content')
    <div class="container">
        <div class="page-header">
            <div class="page-header__back">
                <a href="{{ route('documents.index') }}" class="back-link">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="19" y1="12" x2="5" y2="12"></line>
                        <polyline points="12 19 5 12 12 5"></polyline>
                    </svg>
                    文書一覧へ戻る
                </a>
            </div>

            <h1 class="page-title">{{ $document->filename }}</h1>
            <div class="document-meta">
                <span class="document-meta__item">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="16" y1="2" x2="16" y2="6"></line>
                        <line x1="8" y1="2" x2="8" y2="6"></line>
                        <line x1="3" y1="10" x2="21" y2="10"></line>
                    </svg>
                    {{ $document->created_at->format('Y-m-d H:i') }}
                </span>
                <span class="document-meta__item">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                    {{ strtoupper(pathinfo($document->filename, PATHINFO_EXTENSION)) }}
                </span>
                <span class="document-meta__item document-type">
                    {{ $document->metadata['document_type'] ?? '自動検出' }}
                </span>
            </div>
        </div>

        <div class="summary-container">
            <div class="summary-content">
                <div class="summary-header">
                    <h2 class="summary-title">要約</h2>
                    <div class="summary-actions">
                        <div class="detail-level-switcher">
                            <button type="button"
                                class="summary-detail-btn {{ $document->detail_level == 'brief' ? 'summary-detail-btn--active' : '' }}"
                                data-level="brief">簡潔</button>
                            <button type="button"
                                class="summary-detail-btn {{ $document->detail_level == 'standard' ? 'summary-detail-btn--active' : '' }}"
                                data-level="standard">標準</button>
                            <button type="button"
                                class="summary-detail-btn {{ $document->detail_level == 'detailed' ? 'summary-detail-btn--active' : '' }}"
                                data-level="detailed">詳細</button>
                        </div>
                        <button type="button" class="btn btn--outline btn--icon" title="要約をコピー">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                                stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                        </button>
                        <button type="button" class="btn btn--outline btn--icon" title="PDFとしてダウンロード">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                                stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7 10 12 15 17 10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                        </button>
                    </div>
                </div>

                @if ($document->status == 'processing')
                    <div class="summary-processing">
                        <div class="summary-processing__animation">
                            <div class="loading-spinner"></div>
                        </div>
                        <p class="summary-processing__text">要約処理中です。しばらくお待ちください...</p>
                    </div>
                @elseif ($document->status == 'completed')
                    <div id="summary-content" class="summary-text" data-document-id="{{ $document->id }}"
                        data-keywords="{{ json_encode($document->keywords) }}">
                        {!! $document->summary !!}
                    </div>

                    @if (count($document->keywords) > 0)
                        <div class="keyword-cloud">
                            <h3 class="keyword-cloud__title">重要キーワード</h3>
                            <div class="keyword-cloud__items">
                                @foreach ($document->keywords as $keyword)
                                    <span class="keyword-tag">{{ $keyword }}</span>
                                @endforeach
                            </div>
                        </div>
                    @endif
                @else
                    <div class="summary-error">
                        <p>要約の生成中にエラーが発生しました。再試行してください。</p>
                        <button type="button" class="btn btn--primary">再試行</button>
                    </div>
                @endif
            </div>

            <div class="original-content">
                <h2 class="original-title">原文</h2>
                <div id="original-text" class="original-text">
                    {!! $document->content !!}
                </div>
            </div>
        </div>
    </div>
@endsection

@section('scripts')
    @vite('resources/js/summary.js')
    @if ($document->status == 'processing')
        <script>
            // 5秒ごとにページをリロード（処理状態を更新するため）
            setTimeout(function () {
                location.reload();
            }, 5000);
        </script>
    @endif
@endsection