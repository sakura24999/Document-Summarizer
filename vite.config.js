import { defineConfig } from 'vite';
import laravel from 'laravel-vite-plugin';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
    plugins: [
        laravel({
            input: [
                'resources/css/css/app.scss',
                'resources/css/css/welcome.scss',
                'resources/css/css/_upload.scss', // アップロード用CSSを追加
                'resources/js/app.js',
                'resources/js/upload.js', // アップロード用JSを追加
                'resources/js/summary.js',
            ],
            refresh: true,
        }),
        tailwindcss(),
    ],
});
