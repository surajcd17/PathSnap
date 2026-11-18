"""Call the running API for every listed trip and print a summary.

Run (API must be running):  python scripts/08_api_smoke_test.py [http://<laptop1-ip>:8000]
"""
import sys
import time

import requests

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
print(requests.get(f"{base}/health", timeout=10).json())
trips = requests.get(f"{base}/trips", timeout=30).json()["trips"]
for t in trips:
    t0 = time.time()
    r = requests.post(f"{base}/predict", json={"trip_id": t["trip_id"]}, timeout=600)
    r.raise_for_status()
    j = r.json()
    h, c = j["metrics"].get("hmm", {}), j["metrics"].get("classifier_only", {})
    assert "route" not in j and "reference" not in str(list(j.keys())), "reference data leaked"
    print(f"{t['trip_id']}: obs={j['n_observations']:5d}  clf top1={c.get('top1_acc', float('nan')):.4f}  "
          f"HMM top1={h.get('top1_acc', float('nan')):.4f}  cont={h.get('continuity', float('nan')):.4f}  "
          f"routeF1={h.get('route_f1', float('nan')):.4f}  {time.time() - t0:.2f}s")
