"""
Quick script to check for duplicate videos 
"""

import cv2
import numpy as np
from pathlib import Path
import imagehash
from PIL import Image
from tqdm import tqdm

def get_video_hash(video_path, num_frames=300):
    """Extract perceptual hashes from video frames"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
    
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    hashes = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            hash_val = imagehash.phash(pil_img, hash_size=8)
            hashes.append(hash_val)
    
    cap.release()
    return hashes

def compare_videos(hash1, hash2):
    """Compare two videos and return similarity score (0-1)"""
    if not hash1 or not hash2:
        return 0.0
    
    similarities = []
    for h1, h2 in zip(hash1, hash2):
        hamming_dist = h1 - h2
        similarity = 1 - (hamming_dist / 64.0)
        similarities.append(similarity)
    
    return np.mean(similarities)

def find_duplicates_quick(folder_path, threshold=0.90):
    """Quick duplicate detection"""
    folder = Path(folder_path)
    videos = [
        v for v in list(folder.glob("*.mp4")) + list(folder.glob("*.avi")) +
                     list(folder.glob("*.mov")) + list(folder.glob("*.mkv"))
    ]
    
    print(f"Found {len(videos)} videos in {folder.name}")
    print("Extracting hashes...")
    
    video_hashes = {}
    for video in tqdm(videos):
        try:
            hashes = get_video_hash(video)
            if hashes:
                video_hashes[video] = hashes
        except Exception as e:
            print(f"Error with {video.name}: {e}")
    
    print(f"\nProcessed {len(video_hashes)} videos successfully")
    print("Finding duplicates...")
    
    duplicates = []
    videos_list = list(video_hashes.keys())
    
    for i in tqdm(range(len(videos_list))):
        for j in range(i+1, len(videos_list)):
            sim = compare_videos(video_hashes[videos_list[i]], 
                               video_hashes[videos_list[j]])
            
            if sim >= threshold:
                duplicates.append({
                    'video1': videos_list[i].name,
                    'video2': videos_list[j].name,
                    'similarity': f"{sim:.5%}"
                })
    
    return duplicates


print("="*60)
print("CHECKING SHOPLIFTER FOLDER")
print("="*60)
shop_dups = find_duplicates_quick(r"Shop DataSet\shop lifters", threshold=1)

print("\n" + "="*60)
print("CHECKING NON-SHOPLIFTER FOLDER")
print("="*60)
non_shop_dups = find_duplicates_quick(r"Shop DataSet\non shop lifters", threshold=0.9999)



print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

if shop_dups:
    print(f"\n Found {len(shop_dups)} duplicate pairs in SHOPLIFTER folder:")
    for dup in shop_dups[:]:  # Show first 10
        print(f"  • {dup['video1']} ≈ {dup['video2']} ({dup['similarity']})")
else:
    print("\n No duplicates found in shoplifter folder")

if non_shop_dups:
    print(f"\n Found {len(non_shop_dups)} duplicate pairs in NON-SHOPLIFTER folder:")
    for dup in non_shop_dups[:10]:
        print(f"  • {dup['video1']} ≈ {dup['video2']} ({dup['similarity']})")
else:
    print("\n No duplicates found in non-shoplifter folder")


# we found that in the 'non shop lifters' folder there are 531 - 218 = 313 unique video
#
# there are 218 video repeated 
# shop_lifter_n_0.mp4   and  shop_lifter_n_0_1.mp4 
# shop_lifter_n_1.mp4   and  shop_lifter_n_1_1.mp4 
# shop_lifter_n_2.mp4   and  shop_lifter_n_2_1.mp4 
# shop_lifter_n_3.mp4   and  shop_lifter_n_3_1.mp4 
# shop_lifter_n_4.mp4   and  shop_lifter_n_4_1.mp4 
# ...
# shop_lifter_n_218.mp4   and  shop_lifter_n_218_1.mp4 
# 
# but there is a minssing index -> 171
# after shop_lifter_n_170.mp4   and  shop_lifter_n_170_1.mp4 
# found shop_lifter_n_172.mp4   and  shop_lifter_n_172_1.mp4  
#
# so final there is 218 need to be excluded

