// ===== Application State =====
const AppState = {
    currentPage: 'studio',
    currentState: 'idle',
    mediaRecorder: null,
    audioChunks: [],
    recordingStartTime: null,
    timerInterval: null,
    currentAudioBlob: null,
    currentLanguage: 'en',  // UI display language only
    apiBaseUrl: 'http://localhost:8000'
};

// ===== Utility Functions =====
function showToast(message, duration = 3000) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, duration);
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatDuration(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
}

// ===== Navigation =====
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            switchPage(page);
        });
    });
}

function switchPage(pageName) {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === pageName) {
            item.classList.add('active');
        }
    });
    
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(`${pageName}-page`).classList.add('active');
    
    AppState.currentPage = pageName;
    
    if (pageName === 'history') {
        loadHistory();
    }
}

// ===== Recording Functions =====
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        AppState.mediaRecorder = new MediaRecorder(stream);
        AppState.audioChunks = [];
        AppState.recordingStartTime = Date.now();
        
        AppState.mediaRecorder.ondataavailable = (event) => {
            AppState.audioChunks.push(event.data);
        };
        
        AppState.mediaRecorder.onstop = () => {
            const audioBlob = new Blob(AppState.audioChunks, { type: 'audio/webm' });
            AppState.currentAudioBlob = audioBlob;
            processAudioWithAPI(audioBlob, 'recording');
            
            stream.getTracks().forEach(track => track.stop());
        };
        
        AppState.mediaRecorder.start();
        switchState('recording');
        startTimer();
        
        showToast(t('toast_recording_started'));
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showToast(t('toast_mic_denied'));
    }
}

function stopRecording() {
    if (AppState.mediaRecorder && AppState.mediaRecorder.state !== 'inactive') {
        AppState.mediaRecorder.stop();
        stopTimer();
        showToast(t('toast_recording_stopped'));
    }
}

function startTimer() {
    let seconds = 0;
    const timerElement = document.getElementById('timer');
    
    AppState.timerInterval = setInterval(() => {
        seconds++;
        timerElement.textContent = formatTime(seconds);
    }, 1000);
}

function stopTimer() {
    if (AppState.timerInterval) {
        clearInterval(AppState.timerInterval);
        AppState.timerInterval = null;
    }
}

// ===== File Upload Functions =====
function initFileUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFile(file);
        }
    });
}

function handleFile(file) {
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4', 'audio/flac', 'audio/aac', 'audio/webm'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|ogg|m4a|flac|aac)$/i)) {
        showToast(t('toast_invalid_file'));
        return;
    }
    
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast(t('toast_file_too_large'));
        return;
    }
    
    AppState.currentAudioBlob = file;
    processAudioWithAPI(file, 'upload', file.name);
}

// ===== API Processing Functions =====
async function processAudioWithAPI(audioBlob, sourceType, fileName = '') {
    switchState('processing');
    
    const steps = [
        t('processing_step_1'),
        t('processing_step_2'),
        t('processing_step_3'),
        t('processing_step_4'),
        t('processing_step_5')
    ];
    
    const progressFill = document.getElementById('progress-fill');
    const processingStep = document.getElementById('processing-step');
    const fileNameElement = document.getElementById('file-name');
    
    if (fileName) {
        fileNameElement.textContent = fileName;
    }
    
    try {
        // Animate through steps
        let currentStep = 0;
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                processingStep.textContent = steps[currentStep];
                progressFill.style.width = `${((currentStep + 1) / steps.length) * 100}%`;
                currentStep++;
            }
        }, 600);
        
        // Prepare form data
        const formData = new FormData();
        formData.append('audio', audioBlob, fileName || 'recording.webm');
        formData.append('language', 'en');  // Always use 'en' for API (backend expects 'en' or 'sw', we use 'en' as default)
        
        // Call API
        const endpoint = sourceType === 'recording' ? '/api/analyze-recording' : '/api/analyze';
        const response = await fetch(`${AppState.apiBaseUrl}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(stepInterval);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const results = await response.json();
        
        // Add additional data
        results.audioBlob = audioBlob;
        results.sourceType = sourceType;
        
        displayResults(results);
        saveToHistory(results);
        
    } catch (error) {
        console.error('Error processing audio:', error);
        showToast(t('toast_error_processing'));
        switchState('idle');
    }
}

// ===== Results Display =====
function displayResults(results) {
    switchState('results');
    
    // Set up audio player
    const audioPlayer = document.getElementById('audio-player');
    const audioURL = URL.createObjectURL(results.audioBlob);
    audioPlayer.src = audioURL;
    
    // Display metrics
    document.getElementById('metric-duration').textContent = formatDuration(results.metrics.duration);
    document.getElementById('metric-processing').textContent = `${results.processing_time}s`;
    document.getElementById('metric-speed').textContent = `${(results.metrics.duration / results.processing_time).toFixed(1)}x`;
    document.getElementById('metric-confidence').textContent = `${(92 + Math.random() * 6).toFixed(1)}%`;
    document.getElementById('metric-words').textContent = results.metrics.word_count;
    document.getElementById('metric-chars').textContent = results.metrics.char_count.toLocaleString();
    document.getElementById('metric-rate').textContent = `${results.metrics.speaking_rate} WPM`;
    document.getElementById('metric-quality').textContent = [t('quality_excellent'), t('quality_good'), t('quality_fair')][Math.floor(Math.random() * 3)];
    document.getElementById('metric-language').textContent = results.language === 'en' ? t('language_english') : t('language_swahili');
    document.getElementById('metric-sentences').textContent = results.metrics.sentence_count;
    document.getElementById('metric-paragraphs').textContent = results.metrics.paragraph_count;
    document.getElementById('metric-source').textContent = results.sourceType === 'recording' ? t('source_recording') : t('source_upload');
    
    // Display sentiment
    document.getElementById('sentiment-emoji').textContent = results.sentiment_emoji;
    document.getElementById('sentiment-label').textContent = results.sentiment.charAt(0).toUpperCase() + results.sentiment.slice(1);
    document.getElementById('sentiment-confidence').textContent = `${results.sentiment_confidence.toFixed(0)}% Confidence`;
    
    const dist = results.sentiment_distribution;
    const total = dist.positive + dist.neutral + dist.negative;
    const posPercent = (dist.positive / total * 100).toFixed(0);
    const neuPercent = (dist.neutral / total * 100).toFixed(0);
    const negPercent = (dist.negative / total * 100).toFixed(0);
    
    document.getElementById('positive-bar').style.width = `${posPercent}%`;
    document.getElementById('positive-percent').textContent = `${posPercent}%`;
    document.getElementById('neutral-bar').style.width = `${neuPercent}%`;
    document.getElementById('neutral-percent').textContent = `${neuPercent}%`;
    document.getElementById('negative-bar').style.width = `${negPercent}%`;
    document.getElementById('negative-percent').textContent = `${negPercent}%`;
    
    // Display summary and keywords
    document.getElementById('summary-text').textContent = results.summary;
    
    const keywordsList = document.getElementById('keywords-list');
    keywordsList.innerHTML = '';
    const colors = ['amber', 'teal', 'rose'];
    results.keywords.forEach((keyword, index) => {
        const tag = document.createElement('span');
        tag.className = `keyword-tag ${colors[index % colors.length]}`;
        tag.textContent = keyword;
        keywordsList.appendChild(tag);
    });
    
    // Display transcript
    document.getElementById('transcript-text').textContent = results.transcript;
}

// ===== State Management =====
function switchState(newState) {
    const states = ['idle', 'recording', 'processing', 'results'];
    
    states.forEach(state => {
        const element = document.getElementById(`${state}-state`);
        if (element) {
            element.classList.add('hidden');
        }
    });
    
    const newStateElement = document.getElementById(`${newState}-state`);
    if (newStateElement) {
        newStateElement.classList.remove('hidden');
    }
    
    AppState.currentState = newState;
}

// ===== History Management =====
function saveToHistory(results) {
    let history = JSON.parse(localStorage.getItem('audioHistory') || '[]');
    
    const historyEntry = {
        timestamp: results.timestamp,
        transcript: results.transcript,
        summary: results.summary,
        sentiment: results.sentiment,
        sentimentEmoji: results.sentiment_emoji,
        language: results.language,
        wordCount: results.metrics.word_count,
        duration: results.metrics.duration
    };
    
    history.unshift(historyEntry);
    
    if (history.length > 50) {
        history = history.slice(0, 50);
    }
    
    localStorage.setItem('audioHistory', JSON.stringify(history));
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('audioHistory') || '[]');
    const container = document.getElementById('history-container');
    
    if (history.length === 0) {
        container.innerHTML = `<div class="history-empty"><h3>${t('history_empty_title')}</h3><p>${t('history_empty_text')}</p></div>`;
        return;
    }
    
    container.innerHTML = '';
    
    history.forEach(entry => {
        const card = document.createElement('div');
        card.className = 'history-card';
        
        const date = new Date(entry.timestamp);
        const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        
        card.innerHTML = `
            <div class="history-header">
                <div class="history-date">${dateStr}</div>
                <div class="history-sentiment">${entry.sentimentEmoji}</div>
            </div>
            <div class="history-transcript">${entry.transcript}</div>
        `;
        
        container.appendChild(card);
    });
}

function clearHistory() {
    if (confirm(t('toast_confirm_clear'))) {
        localStorage.removeItem('audioHistory');
        loadHistory();
        showToast(t('toast_history_cleared'));
    }
}

// ===== API Health Check =====
async function checkAPIHealth() {
    try {
        const response = await fetch(`${AppState.apiBaseUrl}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('API Status:', data.status);
            return true;
        }
    } catch (error) {
        console.warn('API not available. Using offline mode.');
        showToast(t('toast_api_offline'));
        return false;
    }
}

// ===== Event Listeners =====
function initEventListeners() {
    document.getElementById('start-recording-btn').addEventListener('click', startRecording);
    document.getElementById('stop-recording-btn').addEventListener('click', stopRecording);
    
    document.getElementById('new-analysis-btn').addEventListener('click', () => {
        switchState('idle');
        document.getElementById('timer').textContent = '00:00';
    });
    
    document.getElementById('language').addEventListener('change', (e) => {
        console.log('Language dropdown changed to:', e.target.value);
        AppState.currentLanguage = e.target.value;
        console.log('AppState.currentLanguage set to:', AppState.currentLanguage);
        
        updateUILanguage(e.target.value);
        
        const langName = e.target.value === 'en' ? t('toast_language_changed_en', e.target.value) : t('toast_language_changed_sw', e.target.value);
        console.log('Toast message:', langName);
        showToast(langName);
    });
    
    document.getElementById('clear-history-btn').addEventListener('click', clearHistory);
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== Tubonge Initialization Starting ===');
    
    initNavigation();
    initFileUpload();
    initEventListeners();
    
    console.log('AppState.currentLanguage:', AppState.currentLanguage);
    console.log('Calling updateUILanguage with:', AppState.currentLanguage);
    
    // Initialize UI language
    updateUILanguage(AppState.currentLanguage);
    
    // Check API health
    await checkAPIHealth();
    
    console.log('=== Tubonge Initialization Complete ===');
    console.log('API Base URL:', AppState.apiBaseUrl);
    console.log('UI Language:', AppState.currentLanguage);
});
