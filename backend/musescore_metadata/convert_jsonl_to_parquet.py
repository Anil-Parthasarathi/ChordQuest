import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent / 'musescore_song_metadata.jsonl'

if not data_path.exists():
    raise FileNotFoundError(f"musescore metadata file not found: {data_path.resolve()}")

try:
    df = pd.read_json(data_path, lines=True)
except Exception as e:
    # Re-raise with context so the caller can log/use it
    raise RuntimeError(f"failed to read musescore metadata ({data_path}): {e}") from e

df.to_parquet(data_path.with_suffix('.parquet'), index=False)