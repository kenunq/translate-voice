import sys
import time
import queue
import torch
import sounddevice as sd
import vosk
from googletrans import Translator


class TranslateVoice:
    """Представление для перевода голоса"""

    LANGUAGE = "en"
    MODEL_ID = "v3_en"
    SAMPLE_RATE = 48000
    SPEAKER = "en_116"
    PUT_ACCENT = True
    PUT_YO = True
    translator = Translator()
    device_torch = torch.device("cpu")
    model_torch, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=LANGUAGE,
        speaker=MODEL_ID,
    )
    samplerate = 44100
    device = None
    # sd.default.device = 'digital output'

    def __init__(self, path_to_model: str):
        self.model_vosk = vosk.Model(path_to_model)
        self.q = queue.Queue()

    def _callback(self, indata, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def start(self):
        """Метод для запуска процесса перевода голоса"""
        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            rec = vosk.KaldiRecognizer(self.model_vosk, self.samplerate)
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    text = rec.Result().split('"')[-2]
                    if text:
                        print(text)
                        self.play_sound(text)

    def play_sound(self, text: str):
        """Метод для перевода и воспроизведения речи"""
        self.model_torch.to(self.device_torch)
        text = self.translator.translate(text, dest=self.LANGUAGE).text
        audio = self.model_torch.apply_tts(
            text=text,
            speaker=self.SPEAKER,
            sample_rate=self.SAMPLE_RATE,
            put_accent=self.PUT_ACCENT,
            put_yo=self.PUT_YO,
        )
        sd.play(audio, self.SAMPLE_RATE)
        time.sleep((len(audio)) / self.SAMPLE_RATE)
        sd.stop()


q1 = TranslateVoice("models/model-small")
q1.start()
