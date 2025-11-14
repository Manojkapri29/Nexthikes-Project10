# scripts/inspect_jsons.py
import json
from pathlib import Path
ROOTS = [Path("data/images/train"), Path("data/labels/train")]
def load(p): 
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print("ERROR reading", p, e); return None

def short(x,n=300):
    s = repr(x).replace("\n"," ")
    return (s[:n]+"...") if len(s)>n else s

for root in ROOTS:
    print("\nINSPECTING", root)
    if not root.exists():
        print("  (not found)")
        continue
    for j in sorted(root.glob("*.json")):
        print("\n----", j.name)
        d = load(j)
        if d is None: continue
        print("Top-level type:", type(d))
        if isinstance(d, dict):
            keys = list(d.keys())[:50]
            print("Keys:", keys)
            # show likely fields
            for cand in ['shapes','regions','objects','annotations','items','instances','labels','imagePath','imageFilename','filename','image','metadata']:
                if cand in d:
                    print(f" Found key '{cand}' (type={type(d[cand])}) sample:", short(d[cand],500))
        elif isinstance(d, list):
            print("List length:", len(d))
            for i,entry in enumerate(d[:5]):
                print(" Entry", i, "type", type(entry), "sample:", short(entry,500))
print("\nINSPECTION COMPLETE.")
