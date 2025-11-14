# debug_annotations.py
import os, cv2, pytesseract, sys
import numpy as np

# === CONFIG ===
IMAGE_DIR = r"data/images/train"     # <-- set this to your images folder (exact path)
LABELS_PATH = r"models/obj.names"    # optional
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # update if needed

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# === helper ===
def list_files(folder):
    imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png','.tiff'))]
    txts = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
    return sorted(imgs), sorted(txts)

def read_names(path):
    if os.path.exists(path):
        with open(path,'r',encoding='utf-8') as f:
            return [l.strip() for l in f if l.strip()]
    return []

def parse_yolo_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
        return (cls,x,y,w,h)
    except:
        return None

def yolo_to_pixel(box, img_w, img_h):
    cls,x,y,w,h = box
    cx = x * img_w; cy = y * img_h
    bw = w * img_w; bh = h * img_h
    x1 = int(round(cx - bw/2)); y1 = int(round(cy - bh/2))
    x2 = int(round(cx + bw/2)); y2 = int(round(cy + bh/2))
    # clamp
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w-1, x2); y2 = min(img_h-1, y2)
    return cls, x1, y1, x2, y2

# === main ===
if not os.path.exists(IMAGE_DIR):
    print("ERROR: IMAGE_DIR not found:", IMAGE_DIR); sys.exit(1)

imgs, txts = list_files(IMAGE_DIR)
print("Found images:", imgs)
print("Found txt files:", txts)
names = read_names(LABELS_PATH)
print("Loaded class names (len={}): {}".format(len(names), names[:10]))

if not imgs:
    print("No images found in folder. Put image files (.jpg/.png) there.")
    sys.exit(0)

for img_file in imgs:
    base = os.path.splitext(img_file)[0]
    img_path = os.path.join(IMAGE_DIR, img_file)
    ann_path = os.path.join(IMAGE_DIR, base + ".txt")
    print("\n--- Processing", img_file)
    print("Image path:", img_path)
    if not os.path.exists(ann_path):
        print("WARNING: No annotation file for this image at:", ann_path)
        continue
    # show first 10 lines of annotation
    with open(ann_path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    print("Annotation lines ({}):".format(len(lines)))
    for i,l in enumerate(lines[:20]):
        print(f" {i+1}: {l!r}")
    # parse
    parsed = []
    for i,l in enumerate(lines):
        p = parse_yolo_line(l)
        if p is None:
            print(f"  -> Line {i+1} is malformed (expected 5 values): {l!r}")
            continue
        cls,x,y,w,h = p
        # warn if values outside [0,1]
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            print(f"  -> Warning: normalized coords out of range on line {i+1}: {p}")
        parsed.append(p)
    if not parsed:
        print("  -> No valid parsed boxes for this image.")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print("  -> ERROR: failed to read image.")
        continue
    h,w = img.shape[:2]
    vis = img.copy()
    out_crops_dir = os.path.join(IMAGE_DIR, base + "_crops_debug")
    os.makedirs(out_crops_dir, exist_ok=True)

    for i,p in enumerate(parsed):
        cls,xn,yn,wn,hn = p
        cls_idx, x1,y1,x2,y2 = yolo_to_pixel(p, w, h)
        if x2 <= x1 or y2 <= y1:
            print(f"  -> Box {i+1} collapsed to zero area after conversion: {x1,y1,x2,y2}")
            continue
        label = names[cls_idx] if cls_idx < len(names) else f"class_{cls_idx}"
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f"{label} ({cls_idx})", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(out_crops_dir, f"crop_{i+1}_{label}.png")
        cv2.imwrite(crop_path, crop)
        # basic OCR on crop
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
        except Exception as e:
            text = f"[Tesseract error: {e}]"
        print(f"  -> Box {i+1}: label={label}, px=({x1},{y1},{x2},{y2}), crop saved: {crop_path}")
        print("      OCR (first 120 chars):", repr(text.strip()[:120]))
    vis_path = os.path.join(IMAGE_DIR, base + "_annotated_debug.png")
    cv2.imwrite(vis_path, vis)
    print("Saved annotated visualization to:", vis_path)
    print("Saved crops to:", out_crops_dir)

print("\nDEBUG complete. Inspect the annotated images and crops above.")
