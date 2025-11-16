import time
import random
import pandas as pd
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from itertools import cycle
import asyncio
import aiohttp
from swiftshadow.classes import ProxyInterface

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get API keys and proxies from environment variables
API_KEYS = [
    os.getenv('LASTFM_API_KEY'),
    os.getenv('LASTFM_API_KEY_2'),
    os.getenv('LASTFM_API_KEY_3')
]

# proxy_manager = ProxyInterface(protocol="http", autoRotate=True)

# Filter out None values
API_KEYS = [key for key in API_KEYS if key]

if not API_KEYS:
    raise ValueError("No LASTFM_API_KEY found in environment variables. Please set at least LASTFM_API_KEY in your .env file.")

# Create a cycle iterator for API keys
api_cycle = cycle(API_KEYS)

print(f"Using {len(API_KEYS)} API key(s) with Webshare rotating proxy")
for i, key in enumerate(API_KEYS, 1):
    print(f"  API key {i}: ...{key[-8:]}")

CHECKPOINT_FILE = "scrape_checkpoint.json"
OUTPUT_CSV = "lastfm_dataset.csv"
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N users

CONCURRENCY_LIMIT = len(API_KEYS)  # Set to number of API keys

def get_next_pair():
    """Get the next API key and the Webshare rotating proxy"""
    api_key = next(api_cycle)
    proxy = # ADD PROXY URL HERE
    return api_key, proxy

def save_checkpoint(processed_users, users_to_process, all_users):
    """Save current progress to checkpoint file"""
    checkpoint = {
        "processed_users": list(processed_users),
        "users_to_process": users_to_process,
        "all_users": list(all_users)
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)
    print(f"  [Checkpoint saved: {len(processed_users)} processed, {len(users_to_process)} in queue]")

def load_checkpoint():
    """Load checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint['processed_users'])} users already processed")
        return (
            set(checkpoint['processed_users']),
            checkpoint['users_to_process'],
            set(checkpoint['all_users'])
        )
    return None, None, None

def append_tracks_to_csv(tracks):
    """Append tracks to CSV file"""
    df = pd.DataFrame(tracks)
    # Write header only if file doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode='w', index=False, header=True)
    else:
        df.to_csv(OUTPUT_CSV, mode='a', index=False, header=False)

async def async_request(url, proxy, session):
    proxies_dict = proxy if proxy else None
    try:
        async with session.get(url, proxy=proxies_dict, timeout=5) as r:
            if r.status == 429:
                print("    Rate limited...")
                await asyncio.sleep(2)  # Backoff
                return None
            return await r.json()
    except Exception as e:
        print(f"    Async request failed: {e}")
        return None

async def get_user_data_async(user, track_limit=100, friend_limit=50):
    api_key1, proxy1 = get_next_pair()
    api_key2, proxy2 = get_next_pair()
    
    tracks_url = f"https://ws.audioscrobbler.com/2.0/?method=user.gettoptracks&user={user}&api_key={api_key1}&format=json&limit={track_limit}"
    friends_url = f"https://ws.audioscrobbler.com/2.0/?method=user.getfriends&user={user}&api_key={api_key2}&format=json&limit={friend_limit}"
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            async_request(tracks_url, proxy1, session),
            async_request(friends_url, proxy2, session)
        ]
        data_tracks, data_friends = await asyncio.gather(*tasks)
    
    tracks = []
    if data_tracks:
        track_list = data_tracks.get("toptracks", {}).get("track", [])
        tracks = [{"user": user, "track": t["name"], "artist": t["artist"]["name"], "playcount": int(t["playcount"])} for t in track_list]
    
    friends = []
    if data_friends:
        friend_list = data_friends.get("friends", {}).get("user", [])
        friends = [friend["name"] for friend in friend_list]
    
    return tracks, friends

# Starting user
seed_users = ["tonariau", "ilakiy", "n0devolucion", "ETDuckQueen", "wordsense", "butterfly2003", "Wannabe Amputee", "darksoulsfan", "piccola lucciola", "SongsForBats", "Vikingfrog86", "Helloomom", "RiotRix", "QuentinXD", "de1der", "marshmallowxpie", "hiatorlo", "Muffin Puffin", "KettilHyde", "NPGLamb", "Geeveecatullus", "mamainhoboken07", "Floodof92", "FormallyMouseCo", "Schizophrenia86", "melodymann", "GavRUlistening", "macdemarco505", "helen-listens", "luna_nova", "bvod", "davorg", "BackOfAPotato", "electrophile888", "clemenceau22", "peanutbudder", "TimeToAct", "Bowthrow", "phukup2", "kaelyndacreator", "heelsintheheart", "naoconvem", "raw_melody_man", "ashen_nemesis", "sniffsy", "Carebear2215", "satelliteofl0ve", "Axozombie", "davedaveman16","Zapador_", "falloutboyliker", "MusicalChknStrp", "bella_not_belle", "withrainfall", "TyDy94", "JazzyCrusader", "maxtonslistens", "nishiomito", "dhorasoo", "Burrpapp", "Starlessred", "vikingfrog86", "Amixor33", "feastofnoise", "Teddeh", "strangeways1987", "Shorty5", "silverhawk79", "SaadSubaru", "Charlatanry", "mattdh12", "Gucci_6an9", "whoozwah", "Lady_peace", "briia-lewiis", "Ace_Reject_", "delukiel", "chiaave"]

# Load checkpoint or initialize
checkpoint_processed, checkpoint_queue, checkpoint_all = load_checkpoint()

if checkpoint_processed is not None:
    processed_users = checkpoint_processed
    users_to_process = checkpoint_queue
    all_users = checkpoint_all
else:
    all_users = set(seed_users)
    processed_users = set()
    users_to_process = list(seed_users)

max_users = 1000000
sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
lock = asyncio.Lock()

async def process_user(user):
    async with sem:
        print(f"Fetching data for {user}...")
        tracks, friends = await get_user_data_async(user)
        
        async with lock:
            if len(tracks) < 100:
                print(f"  -> Not enough tracks found for {user}, skipping.")
                processed_users.add(user)
                return
            
            # Add tracks to dataset (write incrementally to CSV)
            append_tracks_to_csv(tracks)

            # Shuffle friends to ensure diversity
            random.shuffle(friends)
            
            # Add new friends to process (limit to prevent explosion)
            new_friends = [f for f in friends[:5] if f not in all_users] # Limit to 3 new friends to increase diversity
            all_users.update(new_friends)
            users_to_process.extend(new_friends)
            
            # Shuffle the queue to mix users from different social circles
            random.shuffle(users_to_process)
            
            processed_users.add(user)
            print(f"  -> Got {len(tracks)} tracks and {len(friends)} friends")
            
            # Save checkpoint periodically
            if len(processed_users) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(processed_users, users_to_process, all_users)

async def main():
    tasks = []
    while users_to_process and len(processed_users) < max_users:
        async with lock:
            if not users_to_process:
                break
            user = users_to_process.pop(0)
            if user in processed_users:
                continue
        
        task = asyncio.create_task(process_user(user))
        tasks.append(task)
        
        # Minimal delay
        await asyncio.sleep(random.uniform(0.05, 0.15))
    
    await asyncio.gather(*tasks)
    
    # Final checkpoint save
    save_checkpoint(processed_users, users_to_process, all_users)

    # Print summary
    print(f"\nTotal users processed: {len(processed_users)}")
    print(f"Data saved to {OUTPUT_CSV}")
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Total tracks collected: {len(df)}")
        print(df.head())

asyncio.run(main())
