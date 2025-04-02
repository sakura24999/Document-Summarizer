<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Document extends Model
{
    use HasFactory;

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'user_id',
        'filename',
        'path',
        'content',
        'status',
        'detail_level',
        'document_type',
        'options',
        'summary',
        'keywords',
        'summaries',
        'metadata',
    ];

    /**
     * The attributes that should be cast.
     *
     * @var array<string, string>
     */
    protected $casts = [
        'options' => 'array',
        'keywords' => 'array',
        'summaries' => 'array',
        'metadata' => 'array',
    ];

    /**
     * Get the user that owns the document.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get the summaries for the document.
     */
    public function documentSummaries(): HasMany
    {
        return $this->hasMany(DocumentSummary::class);
    }

    /**
     * Check if the document is still being processed.
     */
    public function isProcessing(): bool
    {
        return $this->status === 'processing';
    }

    /**
     * Check if the document processing is completed.
     */
    public function isCompleted(): bool
    {
        return $this->status === 'completed';
    }

    /**
     * Check if the document processing has failed.
     */
    public function hasFailed(): bool
    {
        return $this->status === 'failed';
    }
}
