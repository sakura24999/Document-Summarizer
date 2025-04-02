// resources/js/upload.js
export default class DocumentUploader {
  constructor(dropZoneId, fileInputId, progressBarId) {
    this.dropZone = document.getElementById(dropZoneId);
    this.fileInput = document.getElementById(fileInputId);
    this.progressBar = document.getElementById(progressBarId);
    this.csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

    this.init();
  }

  init() {
    if (!this.dropZone) return;

    // ドラッグ&ドロップイベント
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      this.dropZone.addEventListener(eventName, this.preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      this.dropZone.addEventListener(eventName, () => {
        this.dropZone.classList.add('dropzone--highlight');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      this.dropZone.addEventListener(eventName, () => {
        this.dropZone.classList.remove('dropzone--highlight');
      }, false);
    });

    this.dropZone.addEventListener('drop', this.handleDrop.bind(this), false);
    this.fileInput.addEventListener('change', this.handleFileSelect.bind(this), false);
  }

  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length) {
      this.fileInput.files = files;
      this.uploadFiles(files);
    }
  }

  handleFileSelect(e) {
    const files = e.target.files;
    if (files.length) {
      this.uploadFiles(files);
    }
  }

  uploadFiles(files) {
    const file = files[0]; // 単一ファイルアップロード
    const formData = new FormData();
    formData.append('document', file);

    // プログレスバーの表示
    this.progressBar.style.width = '0%';
    this.progressBar.parentElement.classList.remove('hidden');

    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percentComplete = Math.round((e.loaded / e.total) * 100);
        this.progressBar.style.width = percentComplete + '%';
        this.progressBar.textContent = percentComplete + '%';
      }
    });

    xhr.addEventListener('load', (e) => {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        // アップロード完了後のリダイレクト
        window.location.href = `/documents/${response.document_id}`;
      } else {
        alert('アップロードに失敗しました。');
      }
    });

    xhr.open('POST', '/documents', true);
    xhr.setRequestHeader('X-CSRF-TOKEN', this.csrfToken);
    xhr.send(formData);
  }
}

// 初期化
document.addEventListener('DOMContentLoaded', () => {
  new DocumentUploader('dropzone', 'file-input', 'upload-progress');
});
