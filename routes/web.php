<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\DocumentController;
use App\Http\Controllers\DashboardController;

// 認証関連のルート
Route::middleware(['auth'])->group(function () {
    // ダッシュボード
    Route::get('/dashboard', [DashboardController::class, 'index'])->name('dashboard');

    // ユーザープロフィール
    Route::get('/user/profile', function () {
        return view('profile');
    })->name('user.profile');

    // 文書管理
    Route::prefix('documents')->name('documents.')->group(function () {
        Route::get('/', [DocumentController::class, 'index'])->name('index');
        Route::get('/create', [DocumentController::class, 'create'])->name('create');
        Route::post('/', [DocumentController::class, 'store'])->name('store');
        Route::get('/{document}', [DocumentController::class, 'show'])->name('show');
        Route::delete('/{document}', [DocumentController::class, 'destroy'])->name('destroy');

        // 文書要約関連
        Route::get('/{document}/summary', [DocumentController::class, 'showSummary'])->name('summary');
        Route::match(['get', 'post'], '/{document}/summarize', [DocumentController::class, 'summarize'])->name('summarize');
    });
});

Route::redirect('/', '/register');

Route::redirect('/home', '/dashboard');
