from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch

model = musicgen.MusicGen.get_pretrained('musicgen-medium')
model.set_generation_params(duration=8)

res = model.generate([
    'crazy EDM, heavy bang', 
    'classic reggae track with an electronic guitar solo',
    'lofi slow bpm electro chill with organic samples',
    'rock with saturated guitars, a heavy bass line and crazy drum break and fills.',
    'earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves',
], 
    progress=True)
display_audio(res, 32000)