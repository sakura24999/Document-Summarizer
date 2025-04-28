@extends('layouts.app')

@section('title', 'ダッシュボード')

@section('content')
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>ダッシュボード</h1>
            <div class="dashboard-actions">
                <a href="{{ route('documents.create') }}" class="btn btn-primary">
                    <i class="fas fa-plus"></i> 新規文書アップロード
                </a>
            </div>
        </div>

        <div class="dashboard-stats">
            <div class="stat-card">
                <div class="stat-value">{{ $documentsCount }}</div>
                <div class="stat-label">文書数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ $summariesCount }}</div>
                <div class="stat-label">要約数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ $recentActivityCount }}</div>
                <div class="stat-label">最近の活動</div>
            </div>
        </div>

        <div class="dashboard-sections">
            <div class="dashboard-section">
                <h2>最近の文書</h2>
                @if(count($recentDocuments) > 0)
                    <div class="document-list">
                        @foreach($recentDocuments as $document)
                            <div class="document-item">
                                <div class="document-icon">
                                    <i class="fas fa-{{ $document->getFileTypeIcon() }}"></i>
                                </div>
                                <div class="document-details">
                                    <h3>
                                        <a href="{{ route('documents.show', $document) }}">
                                            {{ $document->title }}
                                        </a>
                                    </h3>
                                    <div class="document-meta">
                                        <span>{{ $document->file_type }}</span>
                                        <span>{{ $document->created_at->diffForHumans() }}</span>
                                    </div>
                                </div>
                                <div class="document-actions">
                                    @if($document->hasSummary())
                                        <a href="{{ route('documents.summary', $document) }}" class="btn btn-sm btn-outline">
                                            <i class="fas fa-eye"></i> 要約を見る
                                        </a>
                                    @else
                                        <a href="{{ route('documents.summarize', $document) }}" class="btn btn-sm btn-outline">
                                            <i class="fas fa-magic"></i> 要約する
                                        </a>
                                    @endif
                                </div>
                            </div>
                        @endforeach
                    </div>
                    <div class="view-all">
                        <a href="{{ route('documents.index') }}">すべての文書を見る</a>
                    </div>
                @else
                    <div class="empty-state">
                        <p>まだ文書がアップロードされていません。</p>
                        <a href="{{ route('documents.create') }}" class="btn btn-primary">文書をアップロード</a>
                    </div>
                @endif
            </div>
        </div>
    </div>
@endsection

@section('scripts')
    <script src="{{ asset('js/dashboard.js') }}"></script>
@endsection