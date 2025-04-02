<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>{{ config('app.name', '専門文書要約アシスタント') }} - @yield('title')</title>

    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">

    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">

    <!-- Styles -->
    <link href="{{ asset('css/app.css') }}" rel="stylesheet">
    @yield('styles')
</head>

<body>
    <div class="sidebar-overlay"></div>

    @auth
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <span>文書要約アシスタント</span>
                </div>
                <div class="sidebar-toggle">
                    <i class="fas fa-bars"></i>
                </div>
            </div>

            <nav class="sidebar-nav">
                <div class="nav-group">
                    <div class="nav-title">メニュー</div>
                    <ul class="nav-items">
                        <li class="nav-item">
                            <a href="{{ route('dashboard') }}"
                                class="{{ request()->routeIs('dashboard') ? 'active' : '' }}">
                                <i class="fas fa-tachometer-alt"></i> ダッシュボード
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="{{ route('documents.index') }}"
                                class="{{ request()->routeIs('documents.*') ? 'active' : '' }}">
                                <i class="fas fa-file-alt"></i> 文書管理
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="{{ route('documents.create') }}"
                                class="{{ request()->routeIs('documents.create') ? 'active' : '' }}">
                                <i class="fas fa-plus"></i> 新規アップロード
                            </a>
                        </li>
                    </ul>
                </div>
            </nav>
        </aside>
    @endauth

    <header class="site-header {{ auth()->check() ? 'with-sidebar' : '' }}">
        <div class="container">
            <div class="site-header__logo">
                <a href="{{ route('dashboard') }}">
                    <h1>専門文書要約アシスタント</h1>
                </a>
            </div>

            <nav class="site-header__nav">
                <ul class="main-nav">
                    @guest
                        <li class="main-nav__item">
                            <a href="{{ route('login') }}" class="main-nav__link">ログイン</a>
                        </li>
                        <li class="main-nav__item">
                            <a href="{{ route('register') }}" class="main-nav__link main-nav__link--highlighted">登録</a>
                        </li>
                    @else
                        <li class="main-nav__item d-md-none">
                            <a href="{{ route('dashboard') }}" class="main-nav__link">ダッシュボード</a>
                        </li>
                        <li class="main-nav__item d-md-none">
                            <a href="{{ route('documents.index') }}" class="main-nav__link">文書一覧</a>
                        </li>
                        <li class="main-nav__item d-md-none">
                            <a href="{{ route('documents.create') }}"
                                class="main-nav__link main-nav__link--highlighted">新規アップロード</a>
                        </li>
                        <li class="main-nav__item main-nav__item--user">
                            <div class="user-dropdown">
                                <div class="user-button">
                                    <div class="avatar">
                                        {{ substr(Auth::user()->name, 0, 1) }}
                                    </div>
                                    <span class="user-name d-none d-md-block">{{ Auth::user()->name }}</span>
                                    <i class="fas fa-chevron-down dropdown-icon"></i>
                                </div>
                                <div class="user-dropdown__menu">
                                    <a href="{{ route('user.profile') }}" class="user-dropdown__link">
                                        <i class="fas fa-user"></i> プロフィール
                                    </a>
                                    <div class="dropdown-divider"></div>
                                    <form method="POST" action="{{ route('logout') }}">
                                        @csrf
                                        <button type="submit" class="user-dropdown__link user-dropdown__link--button">
                                            <i class="fas fa-sign-out-alt"></i> ログアウト
                                        </button>
                                    </form>
                                </div>
                            </div>
                        </li>
                    @endguest
                </ul>
            </nav>
        </div>
    </header>

    <main class="site-content {{ auth()->check() ? 'with-sidebar' : '' }}">
        <div class="container">
            @if (session('status'))
                <div class="alert alert-success">
                    {{ session('status') }}
                </div>
            @endif

            @if (session('error'))
                <div class="alert alert-danger">
                    {{ session('error') }}
                </div>
            @endif

            @yield('content')
        </div>
    </main>

    <footer class="site-footer {{ auth()->check() ? 'with-sidebar' : '' }}">
        <div class="container">
            <p>&copy; {{ date('Y') }} 専門文書要約アシスタント. All rights reserved.</p>
        </div>
    </footer>

    <!-- Scripts -->
    <script src="{{ asset('js/app.js') }}"></script>
    @yield('scripts')

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            // サイドバー切り替え
            const sidebarToggles = document.querySelectorAll('.sidebar-toggle');
            const sidebar = document.querySelector('.sidebar');
            const sidebarOverlay = document.querySelector('.sidebar-overlay');

            if (sidebarToggles.length && sidebar) {
                sidebarToggles.forEach(toggle => {
                    toggle.addEventListener('click', function () {
                        sidebar.classList.toggle('show');
                        if (sidebarOverlay) {
                            sidebarOverlay.classList.toggle('show');
                        }
                    });
                });

                if (sidebarOverlay) {
                    sidebarOverlay.addEventListener('click', function () {
                        sidebar.classList.remove('show');
                        sidebarOverlay.classList.remove('show');
                    });
                }
            }

            // ユーザーメニュードロップダウン
            const userButton = document.querySelector('.user-button');
            const dropdownMenu = document.querySelector('.user-dropdown__menu');

            if (userButton && dropdownMenu) {
                userButton.addEventListener('click', function (event) {
                    event.stopPropagation();
                    userButton.parentElement.classList.toggle('active');
                    dropdownMenu.classList.toggle('show');
                });

                document.addEventListener('click', function (event) {
                    if (!dropdownMenu.contains(event.target) && !userButton.contains(event.target)) {
                        userButton.parentElement.classList.remove('active');
                        dropdownMenu.classList.remove('show');
                    }
                });
            }
        });
    </script>
</body>

</html>