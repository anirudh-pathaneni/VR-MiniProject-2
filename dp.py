import pandas as pd
import os
import glob
from pathlib import Path
import json

# Paths to dataset
dataset_root = "." 
image_metadata_path = os.path.join(dataset_root, "images/metadata/images.csv")
image_dir = os.path.join(dataset_root, "images/small/")
product_metadata_dir = os.path.join(dataset_root, "listings/metadata/")
output_path = os.path.join(dataset_root, "merged_metadata.csv")

# Step 1: Load image metadata
# Contains image_id and path for images in images/small/
try:
    image_metadata = pd.read_csv(image_metadata_path)[["image_id", "path"]]
    print(f"Loaded {len(image_metadata)} image metadata entries")
except FileNotFoundError:
    print(f"Error: {image_metadata_path} not found")
    exit(1)
except Exception as e:
    print(f"Error loading image metadata: {e}")
    exit(1)

# Step 2: Load and combine product metadata from all listings_<i>.json files
product_metadata_files = glob.glob(os.path.join(product_metadata_dir, "listings_*.json"))
if not product_metadata_files:
    print(f"Error: No listings_*.json files found in {product_metadata_dir}")
    exit(1)

product_data = []
for file in product_metadata_files:
    try:
        df = pd.read_json(file, lines=True)
        product_data.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
product_metadata = pd.concat(product_data, ignore_index=True)
print(f"Combined {len(product_metadata)} product metadata entries")

# Step 3: Extract relevant fields from product metadata
def extract_value(field_list, field_name=None):
    """Safely extract 'value' or 'standardized_values' from a list of dictionaries."""
    if not field_list or not isinstance(field_list, list):
        return None
    for item in field_list:
        if isinstance(item, dict):
            if field_name == "color" and "standardized_values" in item and item["standardized_values"]:
                return item["standardized_values"][0]
            if "value" in item:
                return item["value"]
    return None

def extract_item_name(item_names):
    """Format item_name as a semicolon-separated multilingual string."""
    if not item_names or not isinstance(item_names, list):
        return "Unknown"
    return "; ".join(f"{item['language_tag']}: {item['value']}" for item in item_names if isinstance(item, dict) and "language_tag" in item and "value" in item) or "Unknown"

# Apply extraction
product_metadata["item_name"] = product_metadata["item_name"].apply(extract_item_name)
product_metadata["color"] = product_metadata["color"].apply(lambda x: extract_value(x, "color") or "Unknown")
product_metadata["product_type"] = product_metadata["product_type"].apply(
    lambda x: x[0]["value"] if isinstance(x, list) and x and isinstance(x[0], dict) and "value" in x[0] else "Unknown"
)

# Extract and validate bullet_point and item_keywords
product_metadata["bullet_point"] = product_metadata["bullet_point"].apply(
    lambda x: x if isinstance(x, list) and all(isinstance(item, dict) and "language_tag" in item and "value" in item for item in x) else []
)
product_metadata["item_keywords"] = product_metadata["item_keywords"].apply(
    lambda x: list({item["value"]: item for item in x if isinstance(item, dict) and "language_tag" in item and "value" in item}.values()) if isinstance(x, list) else []
)

# Select relevant columns
product_metadata = product_metadata[["item_id", "main_image_id", "other_image_id", "item_name", "color", "product_type", "bullet_point", "item_keywords"]]

# Step 4: Merge with image metadata
# 4.1: Merge on main_image_id
merged_main = image_metadata.merge(
    product_metadata[["item_id", "main_image_id", "item_name", "color", "product_type", "bullet_point", "item_keywords"]],
    left_on="image_id",
    right_on="main_image_id",
    how="inner"
).drop(columns=["main_image_id"])
merged_main["image_source"] = "main"

# 4.2: Merge on other_image_id
product_metadata_exploded = product_metadata.explode("other_image_id")
product_metadata_exploded = product_metadata_exploded[product_metadata_exploded["other_image_id"].notnull() & (product_metadata_exploded["other_image_id"] != "")]
merged_other = image_metadata.merge(
    product_metadata_exploded[["item_id", "other_image_id", "item_name", "color", "product_type", "bullet_point", "item_keywords"]],
    left_on="image_id",
    right_on="other_image_id",
    how="inner"
).drop(columns=["other_image_id"])
merged_other["image_source"] = "other"

# Step 5: Combine merges
merged_data = pd.concat([merged_main, merged_other], ignore_index=True)

# Step 6: Clean up and verify
# Drop duplicates (prioritize main_image_id)
merged_data = merged_data.sort_values("image_source").drop_duplicates(subset=["image_id"], keep="first")
# Add full image path
merged_data["full_image_path"] = merged_data["path"].apply(lambda x: os.path.join(image_dir, x))
# Add image_exists flag
merged_data["image_exists"] = merged_data["full_image_path"].apply(lambda x: Path(x).exists())
# Filter to existing images
merged_data = merged_data[merged_data["image_exists"]]
# Convert bullet_point and item_keywords to JSON strings
merged_data["bullet_point"] = merged_data["bullet_point"].apply(json.dumps)
merged_data["item_keywords"] = merged_data["item_keywords"].apply(json.dumps)
# Drop image_source (not needed for VQA)
merged_data = merged_data[["item_id", "image_id", "item_name", "color", "product_type", "bullet_point", "item_keywords", "full_image_path", "image_exists"]]

# Step 7: Save merged data
merged_data.to_csv(output_path, index=False)
print(f"Merged metadata saved to {output_path} with {len(merged_data)} entries")

# Preview
print("\nSample metadata:")
print(merged_data.head())