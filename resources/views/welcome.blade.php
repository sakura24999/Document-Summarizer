<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>{{ config('app.name', 'Document Summary System') }}</title>

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.bunny.net">
    <link href="https://fonts.bunny.net/css?family=instrument-sans:400,500,600|poppins:300,400,500,600,700"
        rel="stylesheet" />

    <!-- Styles -->
    @vite(['resources/css/css/welcome.scss'])
</head>

<body>
    <!-- 背景アニメーション要素 -->
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>

    <div class="page-wrapper">
        <div class="container">
            <div class="main-content">
                <!-- Header -->
                <header class="header">
                    <div class="welcome-logo-container">
                        <svg class="logo" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M7 18H17V16H7V18Z" fill="currentColor" />
                            <path d="M17 14H7V12H17V14Z" fill="currentColor" />
                            <path d="M7 10H11V8H7V10Z" fill="currentColor" />
                            <path
                                d="M6 2C4.34 2 3 3.34 3 5V19C3 20.66 4.34 22 6 22H18C19.66 22 21 20.66 21 19V9C21 8.4696 20.7893 7.96086 20.4142 7.58579L15.4142 2.58579C15.0391 2.21071 14.5304 2 14 2H6ZM6 4H13V8C13 9.1 13.9 10 15 10H19V19C19 19.55 18.55 20 18 20H6C5.45 20 5 19.55 5 19V5C5 4.45 5.45 4 6 4Z"
                                fill="currentColor" />
                        </svg>
                    </div>
                    <h1 class="welcome-title">AI 文書要約システム</h1>
                    <p class="welcome-subtitle">アップロードするだけで、AIが文書を分析し要約します</p>
                </header>

                <!-- Main Card -->
                <div class="main-card">
                    <!-- Card Body -->
                    <div class="card-content">
                        <!-- Left Section: Feature Illustration -->
                        <div class="card-left">
                            <div class="illustration-container">
                                <div class="illustration-bg"></div>
                                <div class="illustration">
                                    <div class="document-ui">
                                        <div class="document-header">
                                            <div class="document-controls">
                                                <span class="control red"></span>
                                                <span class="control yellow"></span>
                                                <span class="control green"></span>
                                            </div>
                                        </div>
                                        <div class="document-content">
                                            <div class="document-line"></div>
                                            <div class="document-line short"></div>
                                            <div class="document-line"></div>
                                            <div class="document-line short-md"></div>
                                        </div>
                                        <div class="document-summary">
                                            <div class="summary-line"></div>
                                            <div class="summary-line short"></div>
                                        </div>
                                        <div class="document-tags">
                                            <span class="tag">#キーワード</span>
                                            <span class="tag">#抽出</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Right Section: Content -->
                        <div class="card-right">
                            <h2 class="card-title fade-in-delay">
                                文書管理の効率化を実現
                            </h2>
                            <div class="features">
                                <div class="feature slide-right">
                                    <div class="feature-icon">
                                        <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                            xmlns="http://www.w3.org/2000/svg">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12">
                                            </path>
                                        </svg>
                                    </div>
                                    <div class="feature-content">
                                        <h3 class="feature-title">簡単アップロード</h3>
                                        <p class="feature-desc">多様な形式のファイルに対応 (PDF, DOCX, TXT)</p>
                                    </div>
                                </div>
                                <div class="feature slide-right delay-1">
                                    <div class="feature-icon">
                                        <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                            xmlns="http://www.w3.org/2000/svg">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z">
                                            </path>
                                        </svg>
                                    </div>
                                    <div class="feature-content">
                                        <h3 class="feature-title">AI要約</h3>
                                        <p class="feature-desc">高度なAIによる正確かつ簡潔な要約生成</p>
                                    </div>
                                </div>
                                <div class="feature slide-right delay-2">
                                    <div class="feature-icon">
                                        <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                            xmlns="http://www.w3.org/2000/svg">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                                d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z">
                                            </path>
                                        </svg>
                                    </div>
                                    <div class="feature-content">
                                        <h3 class="feature-title">キーワード抽出</h3>
                                        <p class="feature-desc">文書内の重要なキーワードを自動で識別</p>
                                    </div>
                                </div>
                            </div>

                            <!-- CTA Buttons -->
                            <div class="cta-container fade-in-up">
                                @if (Route::has('login'))
                                    @auth
                                        <a href="{{ url('/dashboard') }}" class="btn btn-primary">
                                            ダッシュボード
                                        </a>
                                    @else
                                        <a href="{{ route('login') }}" class="btn btn-primary">
                                            ログイン
                                        </a>
                                        @if (Route::has('register'))
                                            <a href="{{ route('register') }}" class="btn btn-secondary">
                                                新規登録
                                            </a>
                                        @endif
                                    @endauth
                                @endif
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Features Section -->
                <div class="features-grid fade-in-delay-longer">
                    <div class="feature-card">
                        <div class="feature-card-icon purple">
                            <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z">
                                </path>
                            </svg>
                        </div>
                        <h3 class="feature-card-title">セキュアな保存</h3>
                        <p class="feature-card-desc">アップロードした文書と要約結果は安全に保存され、いつでもアクセス可能です。</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-card-icon blue">
                            <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z">
                                </path>
                            </svg>
                        </div>
                        <h3 class="feature-card-title">高コスパ</h3>
                        <p class="feature-card-desc">手動での要約作業を削減し、時間とコストを大幅に節約できます。</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-card-icon green">
                            <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                    d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z">
                                </path>
                            </svg>
                        </div>
                        <h3 class="feature-card-title">高品質な要約</h3>
                        <p class="feature-card-desc">Claudeの高度なAI技術により、人間が作成したかのような質の高い要約を生成します。</p>
                    </div>
                </div>

                <!-- Footer -->
                <footer class="footer">
                    <p>© {{ date('Y') }} 文書要約システム. All rights reserved.</p>
                </footer>
            </div>
        </div>
    </div>
</body>

</html>
