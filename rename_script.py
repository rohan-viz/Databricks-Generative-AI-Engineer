import os
import collections

ROOT_DIR = "/Users/rraj/codes/Databricks-Generative-AI-Engineer"

def rename_files():
    print(f"Scanning directory: {ROOT_DIR}")
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        if ".git" in dirpath:
            continue
            
        # Group files by stem (name without extension)
        stem_map = collections.defaultdict(list)
        for filename in filenames:
            if filename == ".DS_Store" or filename == "README.md":
                continue
            name, ext = os.path.splitext(filename)
            stem_map[name].append(ext)
        
        # Identify collisions
        for stem, extensions in stem_map.items():
            if len(extensions) > 1:
                print(f"Collision found in {dirpath} for stem '{stem}': {extensions}")
                # Rename all colliding files
                # sort extensions to have deterministic order (maybe keep original for one? Plan said all get numbers to be safe/uniform)
                # User asked "prefix 1, 2, 3 etc towards the end". Suffix is what I planned. 
                # Let's do stem + "_1" + ext, stem + "_2" + ext
                
                sorted_exts = sorted(extensions)
                for i, ext in enumerate(sorted_exts):
                    idx = i + 1
                    old_path = os.path.join(dirpath, stem + ext)
                    new_name = f"{stem}_{idx}{ext}"
                    new_path = os.path.join(dirpath, new_name)
                    
                    # Check safety
                    if os.path.exists(new_path) and new_path != old_path:
                        print(f"  SKIPPING: Target {new_name} already exists!")
                        continue
                        
                    print(f"  Renaming: {stem}{ext} -> {new_name}")
                    os.rename(old_path, new_path)

if __name__ == "__main__":
    rename_files()
