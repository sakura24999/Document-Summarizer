<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('documents', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->string('filename');
            $table->string('path');
            $table->text('content')->nullable();
            $table->enum('status', ['processing', 'completed', 'failed'])->default('processing');
            $table->enum('detail_level', ['brief', 'standard', 'detailed'])->default('standard');
            $table->string('document_type')->default('auto');
            $table->json('options')->nullable();
            $table->text('summary')->nullable();
            $table->json('keywords')->nullable();
            $table->json('summaries')->nullable();
            $table->json('metadata')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('documents');
    }
};
