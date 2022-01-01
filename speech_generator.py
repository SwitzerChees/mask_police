try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-text-to-speech-python for
    installation instructions.
    """)
    import sys

    sys.exit(1)

import glob
import random
import pygame.mixer
import os
from dotenv import load_dotenv

pygame.mixer.init()
load_dotenv()
AZURE_KEY = os.getenv('AZURE_KEY')
print(AZURE_KEY)
soundfiles = glob.glob("sounds/*.wav")
speech_key, service_region = AZURE_KEY, "westeurope"
isPlaying = False


def generate_output_speech(name):
    global isPlaying
    if isPlaying is False:
        isPlaying = True
        if name != "Unbekannt":
            output = name.split()
            name = output[0]
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            ssml_string = "<speak version=\"1.0\""
            ssml_string += " xmlns=\"http://www.w3.org/2001/10/synthesis\""
            ssml_string += " xml:lang=\"de-DE\">"
            ssml_string += "<voice name=\"de-CH-LeniNeural\">"
            ssml_string += name + ", zieh deine Maske an!"
            ssml_string += "</voice> </speak>"
            result = synthesizer.speak_ssml_async(ssml_string).get()
            if result.reason == speechsdk.ResultReason.Canceled:
                print("Error:" + str(result.cancellation_details))
        else:
            random_output()
        isPlaying = False


def random_output():
    channel = pygame.mixer.Sound(random.choice(soundfiles)).play()
    while channel.get_busy():
        pygame.time.wait(100)
