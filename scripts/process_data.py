import json
import os

image_dir = "/home/ubuntu/artemis/image_dataset"
image_data = []

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Loop through each category (artwork, nature, urban)
for subdir, _, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('.jpeg', '.jpg')):
            image_path = os.path.join(subdir, file)
            
            # Handling both .jpeg and .jpg for the JSON file
            json_path_jpeg = os.path.join(subdir, file.replace('.jpeg', '.json'))
            json_path_jpg = os.path.join(subdir, file.replace('.jpg', '.json'))

            # Check if JSON file exists for both cases
            json_path = json_path_jpeg if os.path.exists(json_path_jpeg) else json_path_jpg
            
            # Load image metadata (skip loading actual image)
            try:
                metadata = load_json(json_path)
                # Append only the image path and metadata to the image_data list
                image_data.append({
                    'image_path': image_path,  # Storing path instead of image object
                    'metadata': metadata
                })
                print(f"Loaded image: {file}, Metadata: {metadata}")
            except Exception as e:
                print(f"Error loading metadata for {image_path}: {e}")
                continue

# Save image paths and metadata to a JSON file
output_path = 'image_data.json'
with open(output_path, 'w') as f:
    json.dump(image_data, f, indent=4)

print(f"Image data saved to {output_path}")