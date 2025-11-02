import requests
import time
import random
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get API key from environment variable
API_KEY = os.getenv('LASTFM_API_KEY')

if not API_KEY:
    raise ValueError("LASTFM_API_KEY not found in environment variables. Please set it in your .env file.")

def get_user_data(user, track_limit=100, friend_limit=50):
    tracks = []
    friends = []
    
    # Fetch top tracks
    tracks_url = f"https://ws.audioscrobbler.com/2.0/?method=user.gettoptracks&user={user}&api_key={API_KEY}&format=json&limit={track_limit}"
    try:
        r = requests.get(tracks_url, timeout=10)
        data = r.json()
        track_list = data.get("toptracks", {}).get("track", [])
        tracks = [
            {
                "user": user,
                "track": t["name"],
                "artist": t["artist"]["name"],
                "playcount": int(t["playcount"])
            }
            for t in track_list
        ]
    except Exception as e:
        print(f"Error fetching tracks for {user}: {e}")
    
    time.sleep(0.2)
    
    # Fetch friends of this user
    friends_url = f"https://ws.audioscrobbler.com/2.0/?method=user.getfriends&user={user}&api_key={API_KEY}&format=json&limit={friend_limit}"
    try:
        r = requests.get(friends_url, timeout=10)
        data = r.json()
        friend_list = data.get("friends", {}).get("user", [])
        friends = [friend["name"] for friend in friend_list]
    except Exception as e:
        print(f"Error fetching friends for {user}: {e}")
            
    return tracks, friends

# Starting user
seed_users = ["tonariau"]
all_users = set(seed_users)
processed_users = set()
all_data = []

# Crawl through the user network
max_users = 50
users_to_process = list(seed_users)

while users_to_process and len(processed_users) < max_users:
    user = users_to_process.pop(0)
    
    if user in processed_users:
        continue
    
    print(f"Fetching data for {user}...")
    tracks, friends = get_user_data(user)
    
    if len(tracks) < 99:
        print(f"  -> Not enough tracks found for {user}, skipping.")
        processed_users.add(user)
        continue
    
    # Add tracks to dataset
    all_data.extend(tracks)
    
    # Add new friends to process (limit to prevent explosion)
    new_friends = [f for f in friends[:3] if f not in all_users] # Limit to 3 new friends to increase diversity
    all_users.update(new_friends)
    users_to_process.extend(new_friends)
    
    # Shuffle the queue to mix users from different social circles
    random.shuffle(users_to_process)
    
    processed_users.add(user)
    print(f"  -> Got {len(tracks)} tracks and {len(friends)} friends")
    
    # Rate limiting
    time.sleep(random.uniform(0.5, 1.2))

# Save to CSV
print(f"\nTotal tracks collected: {len(all_data)}")
df = pd.DataFrame(all_data)
df.to_csv("lastfm_dataset.csv", index=False)
print(df.head())
