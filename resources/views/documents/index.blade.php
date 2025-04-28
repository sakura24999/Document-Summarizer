@extends('layouts.app')

@section('title', '文書一覧')

@section('content')
    <div class="documents-container">
        <div class="documents-header">
            <h1>文書一覧</h1>
            <div class="documents-actions">
                <a href="{{ route('documents.create') }}" class="btn btn-primary">
                    <i class="fas fa-plus"></i> 新規文書アップロード
                </a>
            </div>
        </div>

        <div class="documents-filters">
            <form action="{{ route('documents.index') }}" method="GET" class="filter-form">
                <div class="form-group">
                    <input type="text" name="search" value="{{ request('search') }}" placeholder="タイトルや内容で検索"
                        class="form-control">
                </div>
                <div class="form-group">
                    <select name="file_type" class="form-control">
                        <option value="">ファイルタイプ</option>
                        <option value="pdf" {{ request('file_type') == 'pdf' ? 'selected' : '' }}>PDF</option>
                        <option value="docx" {{ request('file_type') == 'docx' ? 'selected' : '' }}>DOCX</option>
                        <option value="txt" {{ request('file_type') == 'txt' ? 'selected' : '' }}>TXT</option>
                    </select>
                </div>
                <div class="form-group">
                    <select name="sort" class="form-control">
                        <option value="created_at_desc" {{ request('sort') == 'created_at_desc' ? 'selected' : '' }}>
                            新しい順
                        </option>
                        <option value="created_at_asc" {{ request('sort') == 'created_at_asc' ? 'selected' : '' }}>
                            古い順
                        </option>
                        <option value="title_asc" {{ request('sort') == 'title_asc' ? 'selected' : '' }}>
                            タイトル順
                        </option>
                    </select>
                </div>
                <button type="submit" class="btn btn-outline">フィルター適用</button>
                <a href="{{ route('documents.index') }}" class="btn btn-link">リセット</a>
            </form>
        </div>

        @if(count($documents) > 0)
            <div class="document-list">
                @foreach($documents as $document)
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
                                <form action="{{ route('documents.summarize', $document) }}" method="POST" class="d-inline">
                                    @csrf
                                    <button type="submit" class="btn btn-sm btn-outline">
                                        <i class="fas fa-magic"></i> 要約する
                                    </button>
                                </form>
                            @endif
                            <form action="{{ route('documents.destroy', $document) }}" method="POST" class="d-inline delete-form">
                                @csrf
                                @method('DELETE')
                                <button type="submit" class="btn btn-sm btn-danger">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </form>
                        </div>
                    </div>
                @endforeach
            </div>

            <div class="pagination-container">
                {{ $documents->links() }}
            </div>
        @else
            <div class="empty-state">
                <p>文書が見つかりませんでした。</p>
                <a href="{{ route('documents.create') }}" class="btn btn-primary">文書をアップロード</a>
            </div>
        @endif
    </div>
@endsection

@section('scripts')
    <script>
        document.addEventListener('DOMContentLoaded', function () {
            // 削除確認ダイアログ
            const deleteForms = document.querySelectorAll('.delete-form');
            deleteForms.forEach(form => {
                form.addEventListener('submit', function (e) {
                    if (!confirm('この文書を削除してもよろしいですか？')) {
                        e.preventDefault();
                    }
                });
            });
        });
    </script>
@endsection