import json
import itertools
import random
from pathlib import Path

json_path = "data/annotations/captions_train2017.json"
output_src = "src_train.txt"
output_trg = "trg_train.txt"

def process_coco_for_paraphrasing(json_file, max_pairs_per_image=4):
    print(f"Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Group captions by Image ID
    # Structure: {image_id: ["caption 1", "caption 2", ...]}
    img_to_captions = {}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        caption = ann['caption'].strip().replace('\n', ' ')
        
        if img_id not in img_to_captions:
            img_to_captions[img_id] = []
        img_to_captions[img_id].append(caption)
        
    print(f"Found {len(img_to_captions)} images with captions.")
    
    src_lines = []
    trg_lines = []
    
    count = 0
    
    for img_id, captions in img_to_captions.items():
        # We need at least 2 captions to make a pair
        if len(captions) < 2:
            continue
            
        # Create all possible permutations of length 2
        # e.g., (A, B), (B, A), (A, C), (C, A)...
        pairs = list(itertools.permutations(captions, 2))
        
        # Optional: Limit pairs per image to avoid bias if some images have 20 captions
        if len(pairs) > max_pairs_per_image:
            pairs = random.sample(pairs, max_pairs_per_image)
            
        for src, trg in pairs:
            # Simple cleaning
            if src.lower() == trg.lower(): continue # Skip identicals
            
            src_lines.append(src)
            trg_lines.append(trg)
            count += 1
            
    print(f"Generated {count} paraphrasing pairs.")
    
    return src_lines, trg_lines

def save_pairs(src_list, trg_list, src_file, trg_file):
    print("Saving to disk...")
    with open(src_file, 'w', encoding='utf-8') as f:
        for line in src_list:
            f.write(line + '\n')
            
    with open(trg_file, 'w', encoding='utf-8') as f:
        for line in trg_list:
            f.write(line + '\n')
    print("Done!")
    
if __name__ == "__main__":
    # Ensure you downloaded and unzipped the file first!
    if Path(json_path).exists():
        srcs, trgs = process_coco_for_paraphrasing(json_path)
        save_pairs(srcs, trgs, output_src, output_trg)
        
        # Preview
        print("\n--- Preview ---")
        for i in range(3):
            print(f"Src: {srcs[i]}")
            print(f"Trg: {trgs[i]}")
            print("-" * 20)
    else:
        print(f"Error: File {json_path} not found.")