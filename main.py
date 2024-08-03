import os

from itt import preprocess_image, ocr_core
from tts import text_to_speech_gtts


# Directory containing the images
image_dir = "./real_input"
output_dir = "./real_output"

for dir_ in [image_dir, output_dir]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

# Ensure the directory path is expanded
image_dir = os.path.expanduser(image_dir)


for image_name in sorted(os.listdir(image_dir)):
    if image_name.endswith((".png", ".jpg", ".jpeg")):
        preprocessed_image = preprocess_image(os.path.join(image_dir, image_name))
        text = ocr_core(preprocessed_image)

        print(f"Extracted text from {image_name}: {text}")

        audio_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.wav")
        text_to_speech_gtts(text, audio_file)
