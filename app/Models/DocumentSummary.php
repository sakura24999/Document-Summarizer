<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class DocumentSummary extends Model
{
    use HasFactory;

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'document_id',
        'detail_level',
        'summary',
    ];

    /**
     * Get the document that owns the summary.
     */
    public function document(): BelongsTo
    {
        return $this->belongsTo(Document::class);
    }
}
