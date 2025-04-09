// resources/js/summary.js
export default class SummaryViewer {
  constructor(summaryId, originalTextId) {
    this.summary = document.getElementById(summaryId);
    this.originalText = document.getElementById(originalTextId);
    this.keywords = [];

    this.init();
  }

  init() {
    if (!this.summary || !this.originalText) return;

    // キーワードの取得とハイライト
    this.keywords = JSON.parse(this.summary.dataset.keywords || '[]');
    this.highlightKeywords();

    // 元のテキストと要約の対応箇所へのスクロール機能
    this.setupCorrespondingHighlights();

    // 詳細度切り替えボタンの設定
    const detailBtns = document.querySelectorAll('.summary-detail-btn');
    detailBtns.forEach(btn => {
      btn.addEventListener('click', this.changeDetailLevel.bind(this));
    });
  }

  highlightKeywords() {
    // キーワードのハイライト処理
    if (!this.keywords.length) return;

    let summaryHtml = this.summary.innerHTML;
    let originalHtml = this.originalText.innerHTML;

    this.keywords.forEach(keyword => {
      const regex = new RegExp(`\\b(${keyword})\\b`, 'gi');
      summaryHtml = summaryHtml.replace(regex, '<span class="keyword-highlight">$1</span>');
      originalHtml = originalHtml.replace(regex, '<span class="keyword-highlight" data-keyword="$1">$1</span>');
    });

    this.summary.innerHTML = summaryHtml;
    this.originalText.innerHTML = originalHtml;
  }

  setupCorrespondingHighlights() {
    // 要約と元のテキストの対応箇所をリンク
    const summaryHighlights = this.summary.querySelectorAll('.keyword-highlight');

    summaryHighlights.forEach(highlight => {
      highlight.addEventListener('mouseenter', () => {
        const keyword = highlight.textContent.toLowerCase();
        const originalHighlights = this.originalText.querySelectorAll(`.keyword-highlight[data-keyword="${keyword}"]`);

        originalHighlights.forEach(original => {
          original.classList.add('keyword-highlight--active');

          // スクロール位置の調整
          this.originalText.scrollTop = original.offsetTop - this.originalText.offsetTop - 100;
        });
      });

      highlight.addEventListener('mouseleave', () => {
        const originalHighlights = this.originalText.querySelectorAll('.keyword-highlight--active');
        originalHighlights.forEach(original => {
          original.classList.remove('keyword-highlight--active');
        });
      });
    });
  }

  changeDetailLevel(e) {
    const level = e.target.dataset.level;
    const documentId = this.summary.dataset.documentId;

    // 詳細レベル変更のAPIリクエスト
    fetch(`/api/documents/${documentId}/detail-level`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
      },
      body: JSON.stringify({ level })
    })
    .then(response => response.json())
    .then(data => {
      this.summary.innerHTML = data.summary;
      this.highlightKeywords();
      this.setupCorrespondingHighlights();

      // アクティブボタンのスタイル更新
      document.querySelectorAll('.summary-detail-btn').forEach(btn => {
        btn.classList.remove('summary-detail-btn--active');
      });
      e.target.classList.add('summary-detail-btn--active');
    })
    .catch(error => {
      console.error('詳細レベルの変更に失敗しました:', error);
    });
  }
}

// 再試行処理関数
function handleRetry() {
  const documentId = window.location.pathname.split('/').pop();
  console.log('Retry clicked for document:', documentId);

  // ボタンを無効化して処理中表示
  const retryButton = document.querySelector('.summary-error .btn--primary');
  if (retryButton) {
    retryButton.disabled = true;
    retryButton.innerHTML = '処理中...';
  }

  // エラー表示から処理中表示に変更
  const summaryError = document.querySelector('.summary-error');
  if (summaryError) {
    summaryError.innerHTML = `
      <div class="summary-processing">
        <div class="summary-processing__animation">
          <div class="loading-spinner"></div>
        </div>
        <p class="summary-processing__text">要約処理中です。しばらくお待ちください...</p>
      </div>
    `;
  }

  // 再処理APIリクエスト
  fetch(`/api/documents/${documentId}/retry`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
    }
  })
  .then(response => response.json())
  .then(data => {
    if (data.status === 'success') {
      // 処理開始成功 - 定期的にページをリロード
      setTimeout(function() {
        location.reload();
      }, 5000);
    } else {
      // エラーメッセージを表示
      summaryError.innerHTML = `
        <p>要約の再処理に失敗しました。もう一度お試しください。</p>
        <button type="button" class="btn btn--primary">再試行</button>
      `;

      // 再試行ボタンに再度イベントリスナーを設定
      const newRetryButton = document.querySelector('.summary-error .btn--primary');
      if (newRetryButton) {
        newRetryButton.addEventListener('click', handleRetry);
      }
    }
  })
  .catch(error => {
    console.error('再試行処理でエラーが発生しました:', error);
    summaryError.innerHTML = `
      <p>サーバーとの通信中にエラーが発生しました。もう一度お試しください。</p>
      <button type="button" class="btn btn--primary">再試行</button>
    `;

    // 再試行ボタンに再度イベントリスナーを設定
    const newRetryButton = document.querySelector('.summary-error .btn--primary');
    if (newRetryButton) {
      newRetryButton.addEventListener('click', handleRetry);
    }
  });
}

// ポーリング機能を持つ新しいクラスを追加
class SummaryProcessor {
  constructor(formId, resultContainerId) {
    this.form = document.getElementById(formId);
    this.resultContainer = document.getElementById(resultContainerId);
    this.pollingInterval = null;

    this.init();
  }

  init() {
    if (!this.form) return;

    this.form.addEventListener('submit', this.handleSubmit.bind(this));
  }

  handleSubmit(e) {
    e.preventDefault();
    const formData = new FormData(this.form);

    // 送信ボタンを無効化
    const submitButton = this.form.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    submitButton.innerHTML = '処理中...';

    // 要約リクエスト送信
    fetch(this.form.action, {
      method: 'POST',
      body: formData,
      headers: {
        'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
      }
    })
    .then(response => response.json())
    .then(data => {
      if (data.task_id) {
        // ポーリング開始
        this.startPolling(data.task_id);
      }
    })
    .catch(error => {
      console.error('Error:', error);
      submitButton.disabled = false;
      submitButton.innerHTML = '要約する';
      this.resultContainer.innerHTML = '<div class="alert alert-danger">エラーが発生しました。</div>';
    });
  }

  startPolling(taskId) {
    this.pollingInterval = setInterval(() => {
      fetch(`/api/task/${taskId}`)
      .then(response => response.json())
      .then(data => {
        if (data.status === 'completed') {
          clearInterval(this.pollingInterval);
          this.handleCompletedTask(data);

          // 送信ボタンを再有効化
          const submitButton = this.form.querySelector('button[type="submit"]');
          if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = '要約する';
          }
        } else if (data.status === 'error') {
          clearInterval(this.pollingInterval);
          this.resultContainer.innerHTML = '<div class="alert alert-danger">要約処理中にエラーが発生しました。</div>';

          // 送信ボタンを再有効化
          const submitButton = this.form.querySelector('button[type="submit"]');
          if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = '要約する';
          }
        }
      })
      .catch(error => {
        console.error('Error polling task:', error);
      });
    }, 2000); // 2秒ごとにポーリング
  }

  handleCompletedTask(data) {
    // 要約結果を表示し、ページをリロードするか、または結果を直接表示
    if (data.redirect_url) {
      // 結果ページにリダイレクト
      window.location.href = data.redirect_url;
    } else if (data.result) {
      // 結果を直接表示
      this.resultContainer.innerHTML = `
        <div class="card">
          <div class="card-header">要約結果</div>
          <div class="card-body">
            <div id="summary-content" data-keywords='${JSON.stringify(data.keywords || [])}' data-document-id="${data.document_id}">
              ${data.result}
            </div>
          </div>
        </div>
      `;

      // 元のテキストが既にページにある場合は、SummaryViewerを初期化
      if (document.getElementById('original-text')) {
        new SummaryViewer('summary-content', 'original-text');
      }
    }
  }
}

// 初期化
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded');
  // 既存のSummaryViewerを初期化
  new SummaryViewer('summary-content', 'original-text');

  // 新しいSummaryProcessorを初期化
  new SummaryProcessor('summarize-form', 'summary-result');

  // 再試行ボタンのイベントリスナー設定
  const retryButton = document.querySelector('.summary-error .btn--primary');
  console.log('Retry button found:', retryButton);

  if (retryButton) {
    retryButton.addEventListener('click', handleRetry);
    console.log('Retry event listener attached');
  }
});
