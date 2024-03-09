import torch
import sounddevice as sd
import time

LANGUAGE = 'ru'
MODEL_ID = 'ru_v3'
SAMPLE_RATE = 48000
SPEAKER = 'baya'  # aidar, baya, kseniya, xenia, random
PUT_ACCENT = True
PUT_YO = True

device = torch.device('cpu')
text = 'Рандомный текст.'

model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=LANGUAGE,
                          speaker=MODEL_ID
                          )

model.to(device)

audio = model.apply_tts(text=text, speaker=SPEAKER, sample_rate=SAMPLE_RATE,put_accent=PUT_ACCENT, put_yo=PUT_YO)

print(text)

sd.play(audio, SAMPLE_RATE)
time.sleep(len(audio)/SAMPLE_RATE)
sd.stop()
