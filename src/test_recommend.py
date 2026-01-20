
import random
import numpy as np
from src.recommend import RecommenderSystem

def run_tests():
    print("=== Recommender System Modular Testing ===")
    
    # 1. Initialize System
    try:
        rec_sys = RecommenderSystem(models_dir="models")
    except Exception as e:
        print(f"Error loading RecommenderSystem: {e}")
        return

    # Helper function to print nicely
    def print_recs(user_label, recs):
        print(f"\n[Test: {user_label}]")
        if not recs:
            print("  No recommendations returned.")
            return
        for i, item in enumerate(recs, 1):
            print(f"  {i}. {item['title']} (ID: {item['id']}) | Genres: {item['genres']}")

    # --- TEST 1: New User (Cold Start) ---
    new_user_id = 999999
    recs_new = rec_sys.get_recommendations(user_id=new_user_id, k=5)
    print_recs("Unknown User (Dynamic Cold Start)", recs_new)

    # --- TEST 1.1: Raw Output (Enrich=False) ---
    recs_raw = rec_sys.get_recommendations(user_id=1, k=5, enrich=False)
    print(f"\n[Test: Raw IDs for User 1 (enrich=False)]")
    print(f"  IDs: {recs_raw}")

    # --- TEST 2: Existing User (User 1) ---
    existing_user_id = 1
    recs_existing = rec_sys.get_recommendations(user_id=existing_user_id, k=5)
    print_recs(f"Existing User {existing_user_id}", recs_existing)

    # --- TEST 3: 5 Random Users ---
    # We need a list of valid users to pick from
    # We can get them from the als_artifacts
    all_known_users = list(rec_sys.als_artifacts['user_map'].values())
    random_users = random.sample(all_known_users, 5)
    
    print(f"\n--- Testing 5 Random Users: {random_users} ---")
    for uid in random_users:
        recs_rand = rec_sys.get_recommendations(user_id=uid, k=3)
        print_recs(f"Random User {uid}", recs_rand)

if __name__ == "__main__":
    run_tests()
