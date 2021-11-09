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

speech_key, service_region = "", "westeurope"


def generate_output_speech(name):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat)
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

