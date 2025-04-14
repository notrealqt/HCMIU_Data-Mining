import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Load and combine CSV files into one array
data_1 = pd.read_csv("../game/raw_game_data/steam_game_descriptions_test_20250304_125622_dropAllNull.csv")
data_2 = pd.read_csv("../game/raw_game_data/steam_game_descriptions_test_20250304_222258_dropAllNull.csv")
data = pd.concat([data_1, data_2])
app_ids = data["app_id"][:15000].tolist()  # All app IDs in one array
print(f"Total app IDs to process: {len(app_ids)}")

# Set up a session with retries
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=20)
session.mount("https://", adapter)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_reviews(app_id):
    reviews = []
    try:
        url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&num_per_page=100"
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data["success"] != 1:
            print(f"No reviews available for app ID {app_id}")
            return [{"app_id": app_id, "recommendationid": None, "review": None, "voted_up": None,
                     "timestamp_created": None, "playtime_forever": None,
                     "weighted_vote_score": None, "votes_up": None, "steamid": None}]

        for review in data["reviews"]:
            reviews.append({
                "app_id": app_id,
                "recommendationid": review.get("recommendationid"),
                "review": review.get("review"),
                "voted_up": review.get("voted_up"),
                "timestamp_created": review.get("timestamp_created"),
                "playtime_forever": review["author"].get("playtime_forever"),
                "weighted_vote_score": float(review.get("weighted_vote_score", 0)),
                "votes_up": review.get("votes_up", 0),
                "steamid": review["author"].get("steamid")
            })
        print(f"Success: Fetched {len(reviews)} reviews for app ID {app_id}")
        return reviews

    except requests.exceptions.Timeout as e:
        print(f"Timeout after retries: {app_id} - {e}")
        return [{"app_id": app_id, "recommendationid": None, "review": None, "voted_up": None,
                 "timestamp_created": None, "playtime_forever": None,
                 "weighted_vote_score": None, "votes_up": None, "steamid": None}]
    except requests.exceptions.RequestException as e:
        print(f"Other error: {app_id} - {e}")
        return [{"app_id": app_id, "recommendationid": None, "review": None, "voted_up": None,
                 "timestamp_created": None, "playtime_forever": None,
                 "weighted_vote_score": None, "votes_up": None, "steamid": None}]

# Use ThreadPoolExecutor with max_workers
data = []
max_workers = 3
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_app = {executor.submit(fetch_reviews, app_id): app_id for app_id in app_ids}
    for future in as_completed(future_to_app):
        app_id = future_to_app[future]
        try:
            results = future.result()
            data.extend(results)
        except Exception as e:
            print(f"Exception for app ID {app_id}: {e}")

# Save to CSV with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"steam_game_reviews_all_{timestamp}.csv"
df = pd.DataFrame(data)
df.to_csv(filename, index=False)
print(f"Saved {len(data)} reviews to {filename}")