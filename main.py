import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Step 1: Load the MusicGen model
model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Step 2: Set the generation parameters (duration in seconds)
model.set_generation_params(duration=10)  # generate 10 seconds

# Step 3: Generate the audio sample based on the description
description = ['Pad in A major, one chord, ambient, ethereal, no melody']
wav = model.generate(description)

# Step 4: Save the generated audio
for idx, one_wav in enumerate(wav):
    audio_write(f'{idx}.wav', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

