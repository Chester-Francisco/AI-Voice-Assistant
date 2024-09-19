import os
import assemblyai as aai
from elevenlabs import generate, stream
from openai import OpenAI

aai_key = os.getenv("aai_key")
openai_key = os.getenv("openai_key")
elevenLabs_key = os.getenv("elevenLabs_key")

class AI_Assistant:
    def __init__(self):
        # API Keys for AI's
        aai.settings.api_key = aai_key
        self.openai_client = OpenAI(api_key = openai_key)
        self.elevenlabs_api_key = elevenLabs_key

        self.transcriber = None

        # Prompt
        self.full_transcript = [
            {"role":"system", "content": "You are a receptionist at a doctors office. Be resourceful and efficient in helping the patient."},
        ]

    #Start real-time transcription with assemblyai
    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data = self.on_data,
            on_error = self.on_error,
            on_open = self.on_open,
            on_close = self.on_close,
            end_utterance_silence_threshold = 1000
        )

        # connects to microphone and starts streaming data to AssemblyAI 
        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStrean(sample_rate=16000)
        self.transcriber.stream(microphone_stream)

    # Checks if transcriber is running then closes it and sets it to None
    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    # AssemblyAI Docs Code for transcribing and streaming audio from microphone
    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session ID:", session_opened.session_id)


    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")


    def on_error(self, error: aai.RealtimeError):
        print("An error occured:", error)
        return


    def on_close(self):
        print("Closing Session")
        return


    # generate ai response 
    def generate_ai_response(self, transcript):

        self.stop_transcription()

        self.full_transcript.append({"role":"user", "content": transcript.text})
        print(f'\n Patient: {transcript.text}', end="\r\n")   

        response = self.openai_client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = self.full_transcript
        )

        ai_reponse = response.choices[0].message.content

        self.generate_audio(ai_reponse)

        self.start_transcription()

    # generate audio 
    def generate_audio(self, text):

        self.full_transcript.append({"role":"system", "content": text})
        print(f'\n AI Receptionist: {text}', end="\r\n")

        audio_stream = generate(
            api_key = self.elevenlabs_api_key,
            text = text,
            voice="Alice",
            stream = True
        )

        stream(audio_stream)



greeting = "THank you for calling victoria medical clinic. My name is Sarah, how can I help you today?"


ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()

