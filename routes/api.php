<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\DocumentController;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "api" middleware group. Make something great!
|
*/

Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});

Route::post('/documents/{document}/update-status', [DocumentController::class, 'updateStatus']);

Route::post('/documents/{document}/retry', [DocumentController::class, 'retryProcessing']);

Route::get('/health', function () {
    return response()->json(['status' => 'ok', 'time' => now()->toDateTimeString()]);
});

// テスト用に要約更新を直接呼べるエンドポイントも追加
Route::post('/test-document-update/{document}', function (App\Models\Document $document, Request $request) {
    $document->status = $request->input('status', 'completed');
    $document->summary = $request->input('summary', 'テスト用の要約です。');
    $document->save();

    return response()->json(['success' => true, 'document' => $document]);
});


