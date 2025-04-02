<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>{{ config('app.name', '専門文書要約アシスタント') }} - @yield('title')</title>

    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">

    <!-- Styles -->
    <link href="{{ asset('css/app.css') }}" rel="stylesheet">
    @yield('styles')
</head>

<body>
    <header class="site-header">
        <div class="container">
            <div class="site-header__logo">
                <a href="{{ route('home') }}">
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
                        <li class="main-nav__item">
                            <a href="{{ route('dashboard') }}" class="main-nav__link">ダッシュボード</a>
                        </li>
                        <li class="main-nav__item">
                            <a href="{{ route('documents.index') }}" class="main-nav__link">文書一覧</a>
                        </li>
                        <li class="main-nav__item">
                            <a href="{{ route('documents.create') }}"
                                class="main-nav__link main-nav__link--highlighted">新規アップロード</a>
                        </li>
                        <li class="main-nav__item main-nav__item--user">
                            <div class="user-dropdown">
                                <span>{{ Auth::user()->name }}</span>
                                <div class="user-dropdown__menu">
                                    <a href="{{ route('profile.edit') }}" class="user-dropdown__link">プロフィール</a>
                                    <form method="POST" action="{{ route('logout') }}">
                                        @csrf
                                        <button type="submit"
                                            class="user-dropdown__link user-dropdown__link--button">ログアウト</button>
                                    </form>
                                </div>
                            </div>
                        </li>
                    @endguest
                </ul>
            </nav>
        </div>
    </header>

    <main class="site-content">
        @yield('content')
    </main>

    <footer class="site-footer">
        <div class="container">
            <p>&copy; {{ date('Y') }} 専門文書要約アシスタント. All rights reserved.</p>
        </div>
    </footer>

    <!-- Scripts -->
    <script src="{{ asset('js/app.js') }}"></script>
    @yield('scripts')
</body>

</html>

