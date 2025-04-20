@extends('layouts.app')

@section('title', '文書アップロード')

@section('styles')
    @vite('resources/css/css/_upload.scss')
    @vite('resources/js/upload.js')
@endsection

@section('content')
    <div class="container">
        <div class="page-header">
            <h1 class="page-title">文書アップロード</h1>
            <p class="page-description">要約したい専門文書をアップロードしてください。PDF、DOCX、TXT形式に対応しています。</p>
        </div>

        <div class="upload-container">
            <div id="dropzone" class="dropzone">
                <div class="dropzone__content">
                    <div class="dropzone__icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                    </div>
                    <p class="dropzone__text">ファイルをドラッグ&ドロップするか、<br>クリックしてファイルを選択</p>
                    <p class="dropzone__hint">最大ファイルサイズ: 10MB</p>
                </div>

                <form id="upload-form" action="{{ route('documents.store') }}" method="POST" enctype="multipart/form-data"
                    class="upload-form">
                    @csrf
                    <input type="file" id="file-input" name="document" class="upload-form__file-input"
                        accept=".pdf,.docx,.txt">
                </form>
            </div>

            <div class="upload-options">
                <h3 class="upload-options__title">要約オプション</h3>

                <div class="form-group">
                    <label for="summary-type" class="form-label">文書タイプ</label>
                    <select id="summary-type" name="summary_type" class="form-select">
                        <option value="auto">自動検出</option>
                        <option value="legal">法律文書</option>
                        <option value="technical">技術文書</option>
                        <option value="medical">医療文書</option>
                        <option value="academic">学術論文</option>
                        <option value="business">ビジネス文書</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="summary-level" class="form-label">要約の詳細度</label>
                    <div class="radio-group">
                        <label class="radio-option">
                            <input type="radio" name="detail_level" value="brief" checked>
                            <span class="radio-label">簡潔</span>
                            <span class="radio-hint">要点のみ（元の1/10程度）</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="detail_level" value="standard">
                            <span class="radio-label">標準</span>
                            <span class="radio-hint">主要内容（元の1/5程度）</span>
                        </label>
                        <label class="radio-option">
                            <input type="radio" name="detail_level" value="detailed">
                            <span class="radio-label">詳細</span>
                            <span class="radio-hint">詳細解説（元の1/3程度）</span>
                        </label>
                    </div>
                </div>

                <div class="form-group">
                    <label class="form-checkbox">
                        <input type="checkbox" name="extract_keywords" checked>
                        <span>重要キーワードの抽出</span>
                    </label>
                </div>

                <button type="button" id="upload-button" class="btn btn--primary btn--large">
                    アップロードして要約
                </button>
            </div>

            <div class="upload-progress hidden">
                <div class="progress-bar">
                    <div id="upload-progress" class="progress-bar__fill" style="width: 0%">0%</div>
                </div>
                <p class="upload-status">ファイルをアップロード中...</p>
            </div>
        </div>
    </div>
@endsection

@section('scripts')
    @vite('resources/js/upload.js')
@endsection