// ===== Translations =====
const translations = {
    en: {
        // Navigation
        nav_studio: "Studio",
        nav_history: "History",
        nav_about: "About",
        
        // Page Titles
        page_audio_studio: "Audio Studio",
        page_history: "Analysis History",
        page_about: "About Tubonge",
        
        // Hero Section
        hero_title: "Transform Speech into Insights",
        hero_subtitle: "Record or upload audio for instant transcription and analysis",
        
        // Input Cards
        card_recording_title: "Live Recording",
        card_recording_desc: "Record audio directly from your microphone",
        card_upload_title: "Upload Audio",
        card_upload_desc: "Drag & drop or click to upload audio files",
        btn_start_recording: "Start Recording",
        
        // Drop Zone
        drop_text: "Drop audio file here or click to browse",
        drop_subtext: "Supports MP3, WAV, OGG, M4A, FLAC, AAC (Max 100MB)",
        
        // Features
        feature_transcription_title: "Accurate Transcription",
        feature_transcription_text: "State-of-the-art ASR with 95%+ accuracy",
        feature_sentiment_title: "Sentiment Analysis",
        feature_sentiment_text: "Understand emotional tone and context",
        feature_analytics_title: "Deep Analytics",
        feature_analytics_text: "Comprehensive metrics and insights",
        feature_fast_title: "Lightning Fast",
        feature_fast_text: "Process audio in seconds",
        
        // Recording State
        recording_indicator: "Recording",
        
        // Processing State
        processing_title: "Processing Audio",
        processing_step_1: "Uploading audio to server...",
        processing_step_2: "Analyzing audio quality...",
        processing_step_3: "Running speech recognition...",
        processing_step_4: "Performing sentiment analysis...",
        processing_step_5: "Generating summary...",
        
        // Results State
        results_banner: "Analysis Complete!",
        btn_new_analysis: "New Analysis",
        
        // Metrics
        metric_duration: "Duration",
        metric_processing: "Processing Time",
        metric_speed: "Processing Speed",
        metric_confidence: "ASR Confidence",
        metric_words: "Word Count",
        metric_chars: "Characters",
        metric_rate: "Speaking Rate",
        metric_quality: "Audio Quality",
        metric_language: "Language",
        metric_sentences: "Sentences",
        metric_paragraphs: "Paragraphs",
        metric_source: "Source Type",
        
        // Sentiment
        section_sentiment: "Sentiment Analysis",
        sentiment_positive: "Positive",
        sentiment_neutral: "Neutral",
        sentiment_negative: "Negative",
        sentiment_confidence: "Confidence",
        
        // Summary
        section_summary: "Summary",
        section_keywords: "Key Topics",
        section_transcript: "Full Transcript",
        
        // History
        btn_clear_history: "Clear History",
        history_empty_title: "No analysis history yet",
        history_empty_text: "Your analyzed audio will appear here",
        
        // About
        about_title_1: "Advanced Speech Analytics",
        about_text_1: "Tubonge leverages state-of-the-art AI models to provide accurate speech-to-text transcription and comprehensive audio analysis.",
        about_title_2: "Key Features",
        about_feature_1: "High-accuracy automatic speech recognition (ASR)",
        about_feature_2: "Real-time sentiment analysis",
        about_feature_3: "Comprehensive audio metrics",
        about_feature_4: "Multi-language support (English & Kiswahili)",
        
        // Language Selector
        language_label: "Language:",
        language_english: "English",
        language_swahili: "Kiswahili",
        
        // Source Types
        source_recording: "Live Recording",
        source_upload: "File Upload",
        
        // Quality Levels
        quality_excellent: "Excellent",
        quality_good: "Good",
        quality_fair: "Fair",
        
        // Toast Messages
        toast_recording_started: "Recording started",
        toast_recording_stopped: "Recording stopped",
        toast_mic_denied: "Microphone access denied. Please allow microphone access and try again.",
        toast_invalid_file: "Invalid file type. Please upload an audio file.",
        toast_file_too_large: "File too large. Maximum size is 100MB.",
        toast_error_processing: "Error processing audio. Please try again.",
        toast_language_changed_en: "Language changed to English",
        toast_language_changed_sw: "Language changed to Kiswahili",
        toast_history_cleared: "History cleared",
        toast_api_offline: "API server not connected. Please start the backend server.",
        toast_confirm_clear: "Are you sure you want to clear all history?"
    },
    
    sw: {
        // Navigation
        nav_studio: "Studio",
        nav_history: "Historia",
        nav_about: "Kuhusu",
        
        // Page Titles
        page_audio_studio: "Studio ya Sauti",
        page_history: "Historia ya Uchambuzi",
        page_about: "Kuhusu Tubonge",
        
        // Hero Section
        hero_title: "Badilisha Hotuba kuwa Maarifa",
        hero_subtitle: "Rekodi au pakia sauti kwa tafsiri na uchambuzi wa papo hapo",
        
        // Input Cards
        card_recording_title: "Kurekodi Moja kwa Moja",
        card_recording_desc: "Rekodi sauti moja kwa moja kutoka kwa maikrofoni yako",
        card_upload_title: "Pakia Sauti",
        card_upload_desc: "Buruta na udondoshe au bofya kupakia faili za sauti",
        btn_start_recording: "Anza Kurekodi",
        
        // Drop Zone
        drop_text: "Dondosha faili ya sauti hapa au bofya kuvinjari",
        drop_subtext: "Inasaidia MP3, WAV, OGG, M4A, FLAC, AAC (Upeo 100MB)",
        
        // Features
        feature_transcription_title: "Tafsiri Sahihi",
        feature_transcription_text: "ASR ya kisasa yenye usahihi wa 95%+",
        feature_sentiment_title: "Uchambuzi wa Hisia",
        feature_sentiment_text: "Elewa toni ya kihisia na muktadha",
        feature_analytics_title: "Uchambuzi wa Kina",
        feature_analytics_text: "Vipimo na maarifa kamili",
        feature_fast_title: "Haraka kama Umeme",
        feature_fast_text: "Chakata sauti kwa sekunde",
        
        // Recording State
        recording_indicator: "Inaendelea Kurekodi",
        
        // Processing State
        processing_title: "Inachakata Sauti",
        processing_step_1: "Inapakia sauti kwenye seva...",
        processing_step_2: "Inachambuza ubora wa sauti...",
        processing_step_3: "Inaendesha utambuzi wa hotuba...",
        processing_step_4: "Inafanya uchambuzi wa hisia...",
        processing_step_5: "Inaunda muhtasari...",
        
        // Results State
        results_banner: "Uchambuzi Umekamilika!",
        btn_new_analysis: "Uchambuzi Mpya",
        
        // Metrics
        metric_duration: "Muda",
        metric_processing: "Muda wa Kuchakata",
        metric_speed: "Kasi ya Kuchakata",
        metric_confidence: "Uhakika wa ASR",
        metric_words: "Idadi ya Maneno",
        metric_chars: "Herufi",
        metric_rate: "Kasi ya Kuzungumza",
        metric_quality: "Ubora wa Sauti",
        metric_language: "Lugha",
        metric_sentences: "Sentensi",
        metric_paragraphs: "Aya",
        metric_source: "Aina ya Chanzo",
        
        // Sentiment
        section_sentiment: "Uchambuzi wa Hisia",
        sentiment_positive: "Chanya",
        sentiment_neutral: "Wastani",
        sentiment_negative: "Hasi",
        sentiment_confidence: "Uhakika",
        
        // Summary
        section_summary: "Muhtasari",
        section_keywords: "Mada Muhimu",
        section_transcript: "Nakala Kamili",
        
        // History
        btn_clear_history: "Futa Historia",
        history_empty_title: "Hakuna historia ya uchambuzi bado",
        history_empty_text: "Sauti yako iliyochambulwa itaonekana hapa",
        
        // About
        about_title_1: "Uchambuzi wa Juu wa Hotuba",
        about_text_1: "Tubonge inatumia miundo ya kisasa ya AI kutoa tafsiri sahihi ya hotuba-hadi-maandishi na uchambuzi kamili wa sauti.",
        about_title_2: "Vipengele Muhimu",
        about_feature_1: "Utambuzi wa kiotomatiki wa hotuba wenye usahihi wa juu (ASR)",
        about_feature_2: "Uchambuzi wa hisia wa wakati halisi",
        about_feature_3: "Vipimo kamili vya sauti",
        about_feature_4: "Msaada wa lugha nyingi (Kiingereza na Kiswahili)",
        
        // Language Selector
        language_label: "Lugha:",
        language_english: "Kiingereza",
        language_swahili: "Kiswahili",
        
        // Source Types
        source_recording: "Kurekodi Moja kwa Moja",
        source_upload: "Upakiaji wa Faili",
        
        // Quality Levels
        quality_excellent: "Bora Sana",
        quality_good: "Nzuri",
        quality_fair: "Wastani",
        
        // Toast Messages
        toast_recording_started: "Kurekodi kumeanza",
        toast_recording_stopped: "Kurekodi kumesimamishwa",
        toast_mic_denied: "Ufikiaji wa maikrofoni umekataliwa. Tafadhali ruhusu ufikiaji wa maikrofoni na ujaribu tena.",
        toast_invalid_file: "Aina ya faili si sahihi. Tafadhali pakia faili ya sauti.",
        toast_file_too_large: "Faili ni kubwa sana. Ukubwa wa juu ni 100MB.",
        toast_error_processing: "Hitilafu katika kuchakata sauti. Tafadhali jaribu tena.",
        toast_language_changed_en: "Lugha imebadilishwa kuwa Kiingereza",
        toast_language_changed_sw: "Lugha imebadilishwa kuwa Kiswahili",
        toast_history_cleared: "Historia imefutwa",
        toast_api_offline: "Seva ya API haijaunganishwa. Tafadhali anzisha seva ya nyuma.",
        toast_confirm_clear: "Je, una uhakika unataka kufuta historia yote?"
    }
};

// ===== Translation Function =====
function t(key, lang) {
    // If lang is not provided, try to get it from AppState
    if (!lang) {
        lang = (typeof AppState !== 'undefined' && AppState.currentLanguage) ? AppState.currentLanguage : 'en';
    }
    
    const translation = translations[lang]?.[key] || translations.en[key] || key;
    return translation;
}

// ===== Update UI with Translations =====
function updateUILanguage(lang) {
    console.log('=== updateUILanguage called with lang:', lang, '===');
    
    // Update all elements with data-i18n attribute
    const elements = document.querySelectorAll('[data-i18n]');
    console.log('Found', elements.length, 'elements with data-i18n attribute');
    
    let translatedCount = 0;
    
    elements.forEach((element, index) => {
        const key = element.getAttribute('data-i18n');
        const translation = t(key, lang);
        
        // Only log first 5 for debugging
        if (index < 5) {
            console.log(`  [${index}] ${element.tagName}.${element.className} - ${key} -> ${translation}`);
        }
        
        if (element.tagName === 'INPUT' && element.type === 'button') {
            element.value = translation;
            translatedCount++;
        } else if (element.tagName === 'OPTION') {
            element.textContent = translation;
            translatedCount++;
        } else if (element.tagName === 'BUTTON') {
            // Check if button has a span with data-i18n
            const span = element.querySelector('span[data-i18n]');
            if (span && span.getAttribute('data-i18n') === key) {
                span.textContent = translation;
            } else {
                element.textContent = translation;
            }
            translatedCount++;
        } else {
            element.textContent = translation;
            translatedCount++;
        }
    });
    
    console.log(`Translated ${translatedCount} elements`);
    
    // Update placeholders
    const placeholderElements = document.querySelectorAll('[data-i18n-placeholder]');
    placeholderElements.forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        element.placeholder = t(key, lang);
    });
    console.log(`Updated ${placeholderElements.length} placeholders`);
    
    // Update titles/tooltips
    const titleElements = document.querySelectorAll('[data-i18n-title]');
    titleElements.forEach(element => {
        const key = element.getAttribute('data-i18n-title');
        element.title = t(key, lang);
    });
    console.log(`Updated ${titleElements.length} titles`);
    
    console.log('=== UI language update complete ===');
}
