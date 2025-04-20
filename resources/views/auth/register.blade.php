@extends('layouts.app')

@section('content')
    <div class="form-container">
        <h2 class="form-title">アカウント登録</h2>

        <form method="POST" action="{{ route('register') }}">
            @csrf

            <div class="form-group">
                <label for="name" class="form-label">名前</label>
                <input id="name" type="text" class="form-control @error('name') is-invalid @enderror" name="name"
                    value="{{ old('name') }}" required autofocus>
                @error('name')
                    <span class="invalid-feedback">{{ $message }}</span>
                @enderror
            </div>

            <div class="form-group">
                <label for="email" class="form-label">メールアドレス</label>
                <input id="email" type="email" class="form-control @error('email') is-invalid @enderror" name="email"
                    value="{{ old('email') }}" required>
                @error('email')
                    <span class="invalid-feedback">{{ $message }}</span>
                @enderror
            </div>

            <div class="form-group">
                <label for="password" class="form-label">パスワード</label>
                <input id="password" type="password" class="form-control @error('password') is-invalid @enderror"
                    name="password" required>
                @error('password')
                    <span class="invalid-feedback">{{ $message }}</span>
                @enderror
            </div>

            <div class="form-group">
                <label for="password_confirmation" class="form-label">パスワード（確認）</label>
                <input id="password_confirmation" type="password" class="form-control" name="password_confirmation"
                    required>
            </div>

            <div class="form-actions">
                <button type="submit" class="btn btn-primary">登録</button>
            </div>
        </form>

        <div class="text-center mt-4">
            <p>すでにアカウントをお持ちの方は <a href="{{ route('login') }}">こちら</a> からログインできます。</p>
        </div>
    </div>
@endsection