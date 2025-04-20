# 専門文書要約アシスタント

LaravelとPythonを連携させた、AI駆動の文書要約システム。

## 機能

- 複数形式の文書アップロード (PDF, DOCX, TXT)
- 専門分野に特化した要約生成
- 重要キーワードの抽出
- 要約の詳細度カスタマイズ

## 技術スタック

- フロントエンド: HTML, CSS, JavaScript
- バックエンド: Laravel 10, Python (FastAPI)
- AI: Claude 3.7 API
- インフラ: Docker, MySQL, Redis

## 環境構築

### 前提条件
- Docker と Docker Compose がインストールされていること
- Anthropic API キーを取得していること

### インストール手順

1. リポジトリをクローン
git clone https://github.com/sakura24999/Document-Summarizer

cd document-summarizer

2. 環境変数の設定
cp .env.example .env

3. Docker コンテナのビルドと起動
docker-compose build
docker-compose up -d

4. Laravel の初期設定
docker-compose exec web composer install
docker-compose exec web php artisan key
docker-compose exec web php artisan migrate
docker-compose exec web npm install
docker-compose exec web npm run dev

5. アプリケーションにアクセス
ブラウザで http://localhost:8080 にアクセス

## 開発

### コンテナ内でのコマンド実行

Artisan コマンドの実行
docker-compose exec web php artisan command

composerの実行
docker-compose exec web composer command

npm の実行
docker-compose exec web npm command

### ログの確認
docker-compose logs -f

## ライセンス
MIT

