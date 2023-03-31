import azure.cognitiveservices.speech as speechsdk
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from decouple import config

# Save model
print('Downloading HF models....')
language_id = EncoderClassifier.from_hparams(source="sahita/language-identification", savedir="tmp")
# language_id = EncoderClassifier.from_hparams(source="sahita/language-identification", savedir="tmp", run_opts={"device":"cuda"})   # If inferencing on GPU

# Secrets
SPEECH_KEY = config('SPEECH_KEY', cast=str)
SERVICE_REGION = config('SERVICE_REGION', cast=str)

def speech_language_detection_once_from_file(path):
    
    # Change dict and language list as required
    langID_to_Int={"en-us":0, "hi-in":1, "ta-in":2, "kn-in":3}

    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "hi-IN", "ta-IN", "kn-IN"])


    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)
    audio_config = speechsdk.AudioConfig(filename=path)

    speech_config.enable_audio_logging()
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, value='Latency')
    speech_language_detection = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config, audio_config=audio_config, auto_detect_source_language_config=auto_detect_source_language_config)

    result = speech_language_detection.recognize_once()

    # Check the result. If recognized: appropriate language label. If not recognized: label - 4.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        detected_src_lang = result.properties[
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
        return langID_to_Int[detected_src_lang]
    elif result.reason == speechsdk.ResultReason.NoMatch or result.reason == speechsdk.ResultReason.Canceled:
        return 4


def huggingface_LID_model(path):

    langID_to_Int={"en":0, "hi":1, "other":3}
    signal = language_id.load_audio(path)
    prediction =  language_id.classify_batch(signal)
    return langID_to_Int[prediction[3][0]]