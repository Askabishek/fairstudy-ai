from langdetect import detect, DetectorFactory

# Ensure consistent results
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detects the language of the given text.
    Returns a two-letter ISO 639-1 language code (e.g., 'en', 'hi', 'ta').
    """
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

def get_gtts_lang_code(lang_code: str) -> str:
    """
    Maps detected language codes to gTTS supported language codes.
    gTTS supports a wide range, but some might need mapping (e.g., 'en-us' to 'en').
    This function provides a basic mapping and can be extended.
    """
    # gTTS generally uses ISO 639-1 codes, so direct mapping often works.
    # Handle specific cases or regional variants if necessary.
    if lang_code.startswith("en"):
        return "en"
    elif lang_code.startswith("hi"):
        return "hi"
    elif lang_code.startswith("ta"):
        return "ta"
    # Add more mappings as needed
    return lang_code
