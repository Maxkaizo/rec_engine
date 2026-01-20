
import httpx
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("=== FastAPI Service Integration Testing ===")
    
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # 1. Health Check
        print("\n[Testing GET /]")
        try:
            r = client.get("/")
            print(f"Status: {r.status_code}")
            print(f"Response: {r.json()}")
        except Exception as e:
            print(f"Could not connect to API: {e}")
            return

        # 2. Popular Movies (Cold Start)
        print("\n[Testing GET /popular?k=3]")
        r = client.get("/popular", params={"k": 3})
        print(f"Status: {r.status_code}")
        data = r.json()
        for i, rec in enumerate(data['recommendations'], 1):
            print(f"  {i}. {rec['title']} ({rec['id']})")

        # 3. Personalized Recommendation (Raw IDs)
        user_id = 1
        print(f"\n[Testing GET /recommend/{user_id} (Raw IDs)]")
        r = client.get(f"/recommend/{user_id}", params={"k": 5, "enrich": False})
        print(f"Status: {r.status_code}")
        print(f"IDs: {r.json()['recommendations']}")

        # 4. Personalized Recommendation (Enriched)
        print(f"\n[Testing GET /recommend/{user_id} (Enriched)]")
        r = client.get(f"/recommend/{user_id}", params={"k": 5, "enrich": True})
        print(f"Status: {r.status_code}")
        recs = r.json()['recommendations']
        for i, rec in enumerate(recs, 1):
             print(f"  {i}. {rec['title']} | {rec['genres']}")

        # 5. Error Handling (Non-existent endpoint)
        print("\n[Testing 404 Error]")
        r = client.get("/not-found")
        print(f"Status: {r.status_code} (Expected 404)")

if __name__ == "__main__":
    test_api()
