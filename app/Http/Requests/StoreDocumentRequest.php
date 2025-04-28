<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class StoreDocumentRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return auth()->check();
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'document' => 'required|file|mimes:pdf,docx,txt|max:10240', // 最大 10MB
            'summary_type' => 'nullable|string|in:auto,legal,technical,medical,academic,business',
            'detail_level' => 'nullable|string|in:brief,standard,detailed',
            'extract_keywords' => 'nullable|boolean',
        ];
    }

    /**
     * Get custom messages for validator errors.
     *
     * @return array
     */
    public function messages(): array
    {
        return [
            'document.required' => '文書ファイルを選択してください。',
            'document.file' => 'アップロードされたファイルが無効です。',
            'document.mimes' => '対応しているファイル形式は PDF, DOCX, TXT のみです。',
            'document.max' => 'ファイルサイズは 10MB 以下にしてください。',
            'summary_type.in' => '無効な文書タイプが指定されました。',
            'detail_level.in' => '無効な詳細レベルが指定されました。',
        ];
    }
}
