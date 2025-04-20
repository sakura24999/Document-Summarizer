document.addEventListener('DOMContentLoaded', function() {
    console.log('Upload JS loaded');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const uploadProgress = document.querySelector('.upload-progress');
    const progressBar = document.getElementById('upload-progress');
    const uploadStatus = document.querySelector('.upload-status');

    if (!dropzone || !fileInput || !uploadButton) {
        console.error('Required elements not found');
        return;
    }

    console.log('Upload elements found and initialized');

    // Drag and drop events
    dropzone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropzone.classList.add('active');
    });

    dropzone.addEventListener('dragleave', function() {
        dropzone.classList.remove('active');
    });

    dropzone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropzone.classList.remove('active');

        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            updateDropzoneUI(e.dataTransfer.files[0].name);
        }
    });

    // Click to select file
    dropzone.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (fileInput.files.length) {
            updateDropzoneUI(fileInput.files[0].name);
        }
    });

    // Upload button
    uploadButton.addEventListener('click', function() {
        console.log('Upload button clicked');

        if (!fileInput.files.length) {
            alert('ファイルを選択してください。');
            return;
        }

        // Add form data from options
        const formData = new FormData(uploadForm);
        formData.append('summary_type', document.getElementById('summary-type').value);

        // ラジオボタンの値を取得
        const detailLevel = document.querySelector('input[name="detail_level"]:checked');
        if (detailLevel) {
            formData.append('detail_level', detailLevel.value);
        }

        // チェックボックスの値を取得
        const extractKeywords = document.querySelector('input[name="extract_keywords"]');
        if (extractKeywords) {
            formData.append('extract_keywords', extractKeywords.checked ? '1' : '0');
        }

        // Show progress UI
        uploadProgress.classList.remove('hidden');
        uploadButton.disabled = true;

        // Send AJAX request
        const xhr = new XMLHttpRequest();
        xhr.open('POST', uploadForm.action, true);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');

        // CSRFトークンの追加
        const token = document.querySelector('meta[name="csrf-token"]');
        if (token) {
            xhr.setRequestHeader('X-CSRF-TOKEN', token.getAttribute('content'));
        } else {
            console.error('CSRF token not found');
        }

        // Handle progress
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = percentComplete + '%';
                progressBar.textContent = percentComplete + '%';
            }
        });

        // Handle response
        xhr.onload = function() {
            console.log('XHR response received:', xhr.status, xhr.responseText);

            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.status === 'success') {
                        uploadStatus.textContent = '処理中... リダイレクトします。';
                        window.location.href = '/documents/' + response.document_id;
                    } else {
                        showError(response.message || 'アップロード中にエラーが発生しました。');
                    }
                } catch(e) {
                    showError('レスポンスの解析に失敗しました。');
                    console.error('JSON parse error:', e);
                }
            } else {
                showError('ステータスコード: ' + xhr.status);
            }
        };

        xhr.onerror = function() {
            showError('ネットワークエラーが発生しました。');
            console.error('XHR error');
        };

        console.log('Sending XHR request');
        xhr.send(formData);
    });

    function updateDropzoneUI(filename) {
        const dropzoneContent = dropzone.querySelector('.dropzone__content');
        dropzoneContent.innerHTML = `
            <div class="dropzone__icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
            </div>
            <p class="dropzone__text">${filename}</p>
            <p class="dropzone__hint">クリックして別のファイルを選択</p>
        `;
    }

    function showError(message) {
        uploadStatus.textContent = 'エラー: ' + message;
        uploadStatus.style.color = '#dc3545';
        uploadButton.disabled = false;
        console.error('Error:', message);
    }
});
