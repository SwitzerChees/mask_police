from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat, ResultReason

speech_config = SpeechConfig(subscription="YOURSUBSCRIPTION", region="westeurope")
speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat["Riff24Khz16BitMonoPcm"])
synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)


def generate_output_speech(name):
    ssml_string = "<speak version=\"1.0\""
    ssml_string += " xmlns=\"http://www.w3.org/2001/10/synthesis\""
    ssml_string += " xml:lang=\"de-DE\">"
    ssml_string += "<voice name=\"de-CH-LeniNeural\">"
    ssml_string += name + ", zieh deine Maske an!"
    ssml_string += "</voice> </speak>"
    result = synthesizer.speak_ssml_async(ssml_string).get()
    if result.reason == ResultReason.Canceled:
        print("Error:" + str(result.cancellation_details))
    stream = AudioDataStream(result)
    stream.save_to_wav_file("./file.wav")
