import cv2
import argparse
import csv
import os
import glob
from datetime import datetime

parser = argparse.ArgumentParser(description="Study Room Occupancy Tracker")

parser.add_argument("--image", type=str, required=False, help="Path to a single image file")
parser.add_argument("--dir", type=str, required=False, help="Path to a folder of images") 
parser.add_argument("--capacity", type=int, default=20, help="Maximum capacity of the room")
args = parser.parse_args()
args = parser.parse_args()

def analyze_room(image_path, max_capacity):
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Could not find image at: {image_path}")
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    head_count = len(faces)
    
    occupancy_rate = (head_count / max_capacity) * 100
    
    if occupancy_rate < 50:
        status = "🟢 EMPTY / PLENTY OF SPACE"
    elif occupancy_rate < 90:
        status = "🟡 BUSY / FILLING UP"
    else:
        status = "🔴 FULL"

    print("\n" + "="*45)
    print(" 📊 STUDY ROOM OCCUPANCY REPORT")
    print("="*45)
    print(f" Image Analyzed : {image_path}")
    print(f" Faces Detected : {head_count}")
    print(f" Room Capacity  : {max_capacity}")
    print(f" Occupancy Rate : {occupancy_rate:.1f}%")
    print(f" Status         : {status}")
    print("="*45)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "occupancy_log.csv"
    
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Head Count", "Occupancy Rate"])

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, head_count, f"{occupancy_rate:.1f}%"])
        
    print(f"[INFO] Data successfully logged to {log_file}\n")

if __name__ == "__main__":

    if args.image:
        analyze_room(args.image, args.capacity)
        
    
    elif args.dir:
        print(f"\n[INFO] Scanning directory: {args.dir}")
        
        
        search_path = os.path.join(args.dir, "*.*")
        all_files = glob.glob(search_path)
        
       
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"[ERROR] No images found in the folder: {args.dir}")
        else:
            print(f"[INFO] Found {len(image_files)} images. Processing batch...\n")
            for img_path in image_files:
                analyze_room(img_path, args.capacity)
                
    else:
        print("[ERROR] Please provide either a single image (--image) or a folder (--dir).")