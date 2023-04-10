import shutil

from fastapi import FastAPI, UploadFile

import glob
import time
import json
import azure.cognitiveservices.speech as speechsdk
from decouple import config

# Secrets
SPEECH_KEY = config('SPEECH_KEY', cast=str)
SERVICE_REGION = config('SERVICE_REGION', cast=str)

def speech_language_detection_once_from_continuous(file_path):
    """performs continuous speech language detection with input from an audio file"""
    # <SpeechContinuousLanguageDetectionWithFile>
    # Creates an AutoDetectSourceLanguageConfig, which defines a number of possible spoken languages
    auto_detect_source_language_config = \
        speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=['hi-in', 'en-in', 'ta-in', 'mr-in', 'gu-in', 'kn-in', 'ml-in', 'te-in', 'bn-in'])

    # Creates a SpeechConfig from your speech key and region
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SERVICE_REGION)

    # Set continuous language detection (override the default of "AtStart")
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode, value='Continuous')
    speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "../log_file")

    audio_config = speechsdk.audio.AudioConfig(filename=file_path)

    source_language_recognizer = speechsdk.SourceLanguageRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config)

    done = False
    
    predictions = list()
    
    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        # print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    def audio_recognized(evt: speechsdk.SpeechRecognitionEventArgs):
        """
        callback that catches the recognized result of audio from an event 'evt'.
        :param evt: event listened to catch recognition result.
        :return:
        """
        
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if evt.result.properties.get(
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult) is None:
                print("Unable to detect any language")
            else:
                detected_src_lang = evt.result.properties[
                    speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult]
                json_result = evt.result.properties[speechsdk.PropertyId.SpeechServiceResponse_JsonResult]
                detail_result = json.loads(json_result)
                
                print('\n'*3)
                
                print(detail_result)
                predictions.append((detail_result['PrimaryLanguage']['Language'], detail_result['Offset']))
                
                start_offset = detail_result['Offset']
                duration = detail_result['Duration']
                if duration >= 0:
                    end_offset = duration + start_offset
                else:
                    end_offset = 0
                
                print("Detected language = " + detected_src_lang)
                print(f"Start offset = {start_offset}, End offset = {end_offset}, "
                      f"Duration = {duration} (in units of hundreds of nanoseconds (HNS))")
                global language_detected
                language_detected = True
                

    # Connect callbacks to the events fired by the speech recognizer
    source_language_recognizer.recognized.connect(audio_recognized)
    # source_language_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    # source_language_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    # source_language_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # # stop continuous recognition on either session stopped or canceled events
    source_language_recognizer.session_stopped.connect(stop_cb)
    source_language_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    source_language_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    source_language_recognizer.stop_continuous_recognition()
    # </SpeechContinuousLanguageDetectionWithFile>

    # Connect callbacks to the events fired by the speech recognizer
    source_language_recognizer.recognized.connect(audio_recognized)
    source_language_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    source_language_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    source_language_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    source_language_recognizer.session_stopped.connect(stop_cb)
    source_language_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    source_language_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    source_language_recognizer.stop_continuous_recognition()
    # </SpeechContinuousLanguageDetectionWithFile>
    
    results = [ tuple(predictions[0]) ]
    prev_lang = predictions[0][0]
    for i in predictions[1:]:
        if i[0] == 'unknown':
            continue
        if prev_lang == i[0]:
            prev_lang = i[0]
            continue
        results.append(tuple(i))
        prev_lang = i[0]
    
    return results

app = FastAPI()


@app.post("/predictions/")
async def predict_language(file: UploadFile):
    save_path = f'/tmp/{file.filename}'
    with open(save_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    predictions = speech_language_detection_once_from_continuous(save_path)
    
    return{'predictions':predictions}
    
    
