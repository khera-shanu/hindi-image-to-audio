import re
from typing import List, Tuple
import os
import tempfile

from gtts import gTTS
from pydub import AudioSegment

import torch

from TTS.api import TTS

import soundfile as sf
import numpy as np


# XTTS can only generate text with a maximum of 400 tokens
def split_text_into_batches(text, batch_size=20):
    # Split the text into words
    words = text.split()
    batches = []
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        batches.append(" ".join(batch))

    return batches


def split_by_languages(text: str) -> List[Tuple[str, str]]:
    # Define regex patterns for Hindi and English
    hindi_pattern = r"[\u0900-\u097F\u0981-\u09811]+"
    english_pattern = r"[a-zA-Z]+"

    combined_pattern = f"({hindi_pattern}|{english_pattern})"
    segments = re.findall(combined_pattern, text)

    result = []
    current_lang = None
    current_segment = ""

    for segment in segments:
        if re.match(hindi_pattern, segment):
            lang = "hi"
        else:
            lang = "en"

        if lang != current_lang:
            if current_segment:
                result.append((current_segment.strip(), current_lang))
            current_lang = lang
            current_segment = segment
        else:
            current_segment += " " + segment

    if current_segment:
        result.append((current_segment.strip(), current_lang))

    final_result = []
    for segment, lang in result:
        batches = split_text_into_batches(segment)
        final_result.extend([(batch, lang) for batch in batches])

    return final_result


def text_to_speech_coqui(
    text,
    output_file="output.wav",
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
):
    tts = TTS(model_name=model_name, progress_bar=True, gpu=torch.cuda.is_available())

    segments = split_by_languages(text)

    audio_segments = []
    for segment, lang in segments:
        audio = tts.tts(
            text=segment,
            language=lang,
            speaker_wav=f"samples/input/audio/premium_tts_sample_{lang}.mp3",
        )
        audio_segments.append(audio)
    combined_audio = np.concatenate(audio_segments)
    sf.write(output_file, combined_audio, 24000)
    print(f"Audio saved as {output_file}")


def text_to_speech_gtts(text, output_file="output.mp3", lang="hi"):
    tts = gTTS(text=text.strip(), lang=lang, slow=False)
    tts.save(output_file)
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     sentences = text.split("\n")
    #     audio_segments = []

    #     for sentence in sentences:
    #         if not sentence.strip():
    #             continue

    #         lang = (
    #             "hi"
    #             if any(ord(char) >= 2304 and ord(char) <= 2431 for char in sentence)
    #             else "en"
    #         )

    #         tts = gTTS(text=sentence.strip(), lang=lang, slow=False)
    #         temp_file = os.path.join(tmpdirname, f"temp_{len(audio_segments)}.mp3")
    #         tts.save(temp_file)

    #         audio_segment = AudioSegment.from_mp3(temp_file)
    #         audio_segments.append(audio_segment)

    #     combined_audio = sum(audio_segments)
    #     combined_audio.export(output_file, format="wav")


if __name__ == "__main__":
    with open("samples/output/text/page_1.txt", "r", encoding="utf-8") as file:
        mixed_text = file.read()
        # print(split_by_languages(mixed_text), len(split_by_languages(mixed_text)))
        text_to_speech_gtts(mixed_text, "samples/output/audio/page_1.wav")
