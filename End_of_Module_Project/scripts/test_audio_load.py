import pandas as pd
from pathlib import Path
import librosa

# Load test data
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
test_df = pd.read_csv(DATA_DIR / 'test.csv')

# Test first sample
row = test_df.iloc[0]
print(f"Audio column type: {type(row['audio'])}")
print(f"Audio value: {row['audio'][:100] if isinstance(row['audio'], str) else row['audio']}")

# Try to extract path
if isinstance(row['audio'], str):
    import ast
    audio_dict = ast.literal_eval(row['audio'])
    audio_path = audio_dict['path']
    print(f"\nAudio path: {audio_path}")
    print(f"File exists: {Path(audio_path).exists()}")
    
    # Load audio
    speech, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio shape: {speech.shape}")
    print(f"Sample rate: {sr}")
    print(f"Duration: {len(speech)/sr:.2f}s")
