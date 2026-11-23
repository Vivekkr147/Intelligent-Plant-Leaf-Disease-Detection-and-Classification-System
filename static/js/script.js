/**
 * =============================================================================
 * PlantCare AI — Client-Side Interaction & Upload Validation
 * Author: vivek kumar (MCA Minor Project)
 * =============================================================================
 * Description:
 * Manages modern drag-and-drop interactions, real-time client-side file
 * validation (size <= 5MB, format in JPG/JPEG/PNG), live photo previews using
 * the HTML5 FileReader API, and friendly user feedback states.
 * =============================================================================
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------------------
    // DOM Element References
    // -------------------------------------------------------------------------
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadForm = document.getElementById('leafUploadForm');

    const previewContainer = document.getElementById('clientPreviewContainer');
    const previewImg = document.getElementById('clientPreviewImg');
    const clientFileName = document.getElementById('clientFileName');
    const clientFileSize = document.getElementById('clientFileSize');
    const clientFileType = document.getElementById('clientFileType');
    const removeFileBtn = document.getElementById('removeFileBtn');

    const errorAlert = document.getElementById('clientErrorAlert');
    const errorMessage = document.getElementById('clientErrorMessage');

    const submitBtn = document.getElementById('submitBtn');
    const submitBtnText = document.getElementById('submitBtnText');
    const submitSpinner = document.getElementById('submitSpinner');

    // Maximum allowed file size: 5 Megabytes in bytes
    const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024;
    const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png'];

    // Exit early if we are not on the upload/predict page
    if (!dropZone || !fileInput) {
        return;
    }

    // -------------------------------------------------------------------------
    // Helper Functions
    // -------------------------------------------------------------------------
    /**
     * Show a client-side warning/error message.
     * @param {string} msg - The error description to show.
     */
    function showError(msg) {
        if (errorAlert && errorMessage) {
            errorMessage.textContent = msg;
            errorAlert.classList.remove('d-none');
            // Auto scroll error into view gently if offscreen
            errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    /**
     * Hide client-side error alert.
     */
    function clearError() {
        if (errorAlert) {
            errorAlert.classList.add('d-none');
        }
    }

    /**
     * Convert bytes into human-readable string (KB or MB).
     * @param {number} bytes 
     * @returns {string} Formatted size string
     */
    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = 2;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    /**
     * Clear selected file and reset preview container.
     */
    function resetSelection() {
        fileInput.value = '';
        if (previewContainer) {
            previewContainer.classList.add('d-none');
        }
        if (previewImg) {
            previewImg.src = '#';
        }
        clearError();
    }

    /**
     * Validate and process a user-selected File object.
     * @param {File} file 
     */
    function processSelectedFile(file) {
        clearError();

        if (!file) {
            return;
        }

        // 1. Validate File Extension
        const fileName = file.name;
        const fileExt = fileName.split('.').pop().toLowerCase();

        if (!ALLOWED_EXTENSIONS.includes(fileExt)) {
            showError(`Unsupported file format (.${fileExt}). Please select a standard JPG, JPEG, or PNG leaf photo.`);
            resetSelection();
            return;
        }

        // 2. Validate File Size limit (5 MB)
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError(`File size (${formatBytes(file.size)}) exceeds the maximum allowed limit of 5 MB. Please choose a smaller image.`);
            resetSelection();
            return;
        }

        // 3. Ensure file has positive size
        if (file.size === 0) {
            showError('The selected file appears to be empty (0 bytes). Please choose an authentic image file.');
            resetSelection();
            return;
        }

        // 4. Update preview metadata fields
        if (clientFileName) clientFileName.textContent = fileName;
        if (clientFileSize) clientFileSize.textContent = formatBytes(file.size);
        if (clientFileType) clientFileType.textContent = fileExt.toUpperCase();

        // 5. Read image into HTML5 FileReader for instant client-side preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) {
                previewImg.src = e.target.result;
            }
            if (previewContainer) {
                previewContainer.classList.remove('d-none');
            }
        };
        reader.onerror = () => {
            showError('Could not read the selected file. Please select a valid photo.');
            resetSelection();
        };
        reader.readAsDataURL(file);
    }

    // -------------------------------------------------------------------------
    // Event Listeners: Browse Button & Direct Dropzone Click
    // -------------------------------------------------------------------------
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent bubbling up to dropZone
            fileInput.click();
        });
    }

    dropZone.addEventListener('click', (e) => {
        // Only trigger file click if not clicking browse button or inner interactive controls
        if (e.target !== browseBtn && !browseBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    // Keyboard accessibility for dropzone (Enter or Space key)
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // File Input change event
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            processSelectedFile(e.target.files[0]);
        }
    });

    // Remove / Clear Button
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', (e) => {
            e.preventDefault();
            resetSelection();
        });
    }

    // -------------------------------------------------------------------------
    // Drag and Drop Event Listeners
    // -------------------------------------------------------------------------
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-active');
        });
    });

    ['dragleave', 'dragend'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-active');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-active');

        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const droppedFile = e.dataTransfer.files[0];
            // Assign dropped file to fileInput so standard form submit works seamlessly
            fileInput.files = e.dataTransfer.files;
            processSelectedFile(droppedFile);
        }
    });

    // -------------------------------------------------------------------------
    // Form Submission & UX Spinner State
    // -------------------------------------------------------------------------
    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            // Check if file has been selected
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                showError('Please choose or drop a plant leaf image before submitting.');
                return;
            }

            // Verify size once more before posting to server
            if (fileInput.files[0].size > MAX_FILE_SIZE_BYTES) {
                e.preventDefault();
                showError('Image exceeds the 5 MB maximum size limit. Please upload a smaller image.');
                return;
            }

            // Provide visual loading feedback while Pillow inspects the image
            if (submitBtn && submitSpinner && submitBtnText) {
                submitBtn.disabled = true;
                submitSpinner.classList.remove('d-none');
                submitBtnText.textContent = 'Validating Leaf Image...';
            }
        });
    }
});
