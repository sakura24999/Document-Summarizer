<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Summary extends Model
{
    use HasFactory;

    protected $fillable = [
        'document_id',
        'content',
        'keywords',
        'highlights'
    ];

    protected $casts = [
        'keywords' => 'array',
    ];

    /**
     * 関連するドキュメントを取得
     */
    public function document()
    {
        return $this->belongsTo(Document::class);
    }
}
