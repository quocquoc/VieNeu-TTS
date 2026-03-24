import sys; sys.path.insert(0, 'src')
from vieneu import Vieneu
import soundfile as sf

# Test with ORIGINAL base model (not your fine-tuned one)
tts = Vieneu(backbone_repo='pnnbao-ump/VieNeu-TTS-0.3B', backbone_device='cuda', codec_device='cuda')
audio = tts.infer('Xin chào, đây là bài kiểm tra.')
sf.write('test_base_model.wav', audio, 24000)
print(f'Base model: {len(audio)} samples, {len(audio)/24000:.1f}s')

# Now test with YOUR fine-tuned model
tts2 = Vieneu(backbone_repo='finetune/output/merged_model', backbone_device='cuda', codec_device='cuda')
voice = tts2.get_preset_voice()
audio2 = tts2.infer('Xin chào, đây là bài kiểm tra.', voice=voice, temperature=0.8)
sf.write('test_finetuned_model.wav', audio2, 24000)
print(f'Fine-tuned: {len(audio2)} samples, {len(audio2)/24000:.1f}s')
