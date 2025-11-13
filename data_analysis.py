import pandas as pd

df = pd.read_csv("lastfm_dataset.csv")
unique_tracks = df['track'].nunique()
unique_artists = df['artist'].nunique()
print(f"Unique tracks: {unique_tracks:,}")
print(f"Unique artists: {unique_artists:,}")
