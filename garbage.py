import os

# Update these with your actual paths
txt_folder = r"/Users/emanuelerimoldi/Desktop/Trial/augmented_data/train/labels"
img_folder = r"/Users/emanuelerimoldi/Desktop/Trial/augmented_data/train/images"

for filename in os.listdir(txt_folder):
    if filename.startswith("Choco_") and filename.endswith(".txt"):
        txt_path = os.path.join(txt_folder, filename)
        if os.path.getsize(txt_path) == 0:
            # Delete the .txt file
            os.remove(txt_path)
            print(f"Deleted empty file: {txt_path}")

            # Construct corresponding .png file path
            base_name = os.path.splitext(filename)[0]  # choco_XXXXXX
            png_path = os.path.join(img_folder, f"{base_name}.png")

            # Delete .png file if it exists
            if os.path.exists(png_path):
                os.remove(png_path)
                print(f"Deleted corresponding image: {png_path}")