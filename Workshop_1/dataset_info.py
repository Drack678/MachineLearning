import os
import hashlib

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    data_dir = "data/raw/"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found. Please download the dataset first.")
        return

    files = os.listdir(data_dir)
    if not files:
        print(f"No files found in {data_dir}")
        return

    print("--- Dataset Information Summary ---")
    for file in files:
        path = os.path.join(data_dir, file)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        file_hash = get_file_hash(path)

        print(f"File: {file}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  SHA256: {file_hash}")

        # This part depends on the specific text format
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                words = " ".join(lines).split()
                unique_words = set(words)
                print(f"  Lines: {len(lines)}")
                print(f"  Tokens: {len(words)}")
                print(f"  Types: {len(unique_words)}")
                print(f"  TTR (Type-Token Ratio): {len(unique_words)/len(words):.4f}")
        except Exception as e:
            print(f"  Could not process file for text statistics: {e}")
        print("-" * 30)

if __name__ == "__main__":
    main()
