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

// 初期化
document.addEventListener('DOMContentLoaded', () => {
  new SummaryViewer('summary-content', 'original-text');
});
