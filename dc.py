import pandas as pd
import os
from google import genai
from PIL import Image
import time
import json

dataset_root = "."
metadata_path = os.path.join(dataset_root, "merged_metadata.csv")
output_prefix = os.path.join(dataset_root, "vqa_dataset")

# Configure Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY environment variable")
client = genai.Client()

# Load merged metadata
merged_data = pd.read_csv(metadata_path)
print(f"Loaded {len(merged_data)} metadata entries")

# Generate VQA questions for a batch
def generate_vqa_batch(start_idx, end_idx):
    vqa_data = []
    subset_data = merged_data[merged_data["image_exists"]].iloc[start_idx:end_idx]
    print(f"Processing images {start_idx} to {end_idx} ({len(subset_data)} images)")

    for _, row in subset_data.iterrows():
        image_path = row["full_image_path"]
        item_name = row["item_name"]
        color = row["color"] if pd.notnull(row["color"]) else "Unknown"
        product_type = row["product_type"]
        
        # Parse bullet_point and item_keywords
        try:
            bullet_point = json.loads(row["bullet_point"]) if pd.notnull(row["bullet_point"]) else []
            item_keywords = json.loads(row["item_keywords"]) if pd.notnull(row["item_keywords"]) else []
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse bullet_point or item_keywords for image {image_path}")
            bullet_point = []
            item_keywords = []

        # Format bullet points (limit to top 3 per language, prioritize en_US)
        bullet_text = []
        languages = set(bp["language_tag"] for bp in bullet_point)
        primary_lang = "en_US" if "en_US" in languages else languages.pop() if languages else None
        secondary_lang = next((lang for lang in languages if lang != primary_lang), None)
        
        if primary_lang:
            primary_bullets = [bp["value"] for bp in bullet_point if bp["language_tag"] == primary_lang][:3]
            bullet_text.extend([f"{primary_lang}: {b}" for b in primary_bullets])
        if secondary_lang:
            secondary_bullets = [bp["value"] for bp in bullet_point if bp["language_tag"] == secondary_lang][:3]
            bullet_text.extend([f"{secondary_lang}: {b}" for b in secondary_bullets])

        # Format item keywords (deduplicate, prioritize en_US)
        keywords = list(set(kw["value"] for kw in item_keywords if kw["language_tag"] == "en_US"))
        if not keywords and item_keywords:
            keywords = list(set(kw["value"] for kw in item_keywords))[:10]

        # Prepare prompt
        prompt = (
            f"Generate exactly 3 Visual Question Answering (VQA) questions with single-word answers for a multimodal VQA dataset."
            f"Questions must be such that one can answer just by looking at the image, only using metadata for guidance or confirmation."
            f"Each question must have 4 multiple-choice options (the correct answer plus 3 plausible distractors). "
            f"Return the response in strictly valid JSON format with fields: question, answer, options (list of 4 strings).\n\n"
            f"Metadata:\n"
            f"- Item Name: {item_name}\n"
            f"- Color: {color}\n"
            f"- Product Type: {product_type}\n"
        )

        # Send API request
        try:
            image = Image.open(image_path)
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite-001",
                contents=[prompt, image]
            )

            # Validate response
            questions = []
            if response is None or not hasattr(response, "text") or response.text is None:
                print(f"Warning: No valid response for image {image_path}")
            else:
                response_text = response.text.strip()
                # Strip Markdown
                if response_text.startswith("```json"):
                    response_text = response_text[7:].rstrip("```").strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:].rstrip("```").strip()
                # Parse JSON
                try:
                    questions = json.loads(response_text)
                except json.JSONDecodeError:
                    print(f"Warning: Non-JSON response for image {image_path}: {response_text}")

            # Process API questions (up to 3)
            valid_api_questions = []
            for q in questions:
                if not (isinstance(q, dict) and "question" in q and "answer" in q and "options" in q and q["question"] and q["answer"] and isinstance(q["options"], list) and len(q["options"]) == 4):
                    print(f"Skipped question for image {image_path}: {q}")
                    continue
                if q["answer"] not in q["options"]:
                    print(f"Warning: Answer '{q['answer']}' not in options {q['options']} for image {image_path}")
                    continue
                valid_api_questions.append({
                    "image_id": row["image_id"],
                    "question": q["question"],
                    "answer": q["answer"],
                    "options": q["options"]
                })

            # Store up to 3 questions
            num_api_questions = len(valid_api_questions)
            questions_to_store = valid_api_questions[:3]
            remaining_slots = 3 - num_api_questions

            if remaining_slots > 0 and pd.notnull(color):
                questions_to_store.append({
                    "image_id": row["image_id"],
                    "question": f"What is the color of the {product_type.lower()}?",
                    "answer": color,
                    "options": [color, "Blue", "Green", "Black"]
                })
                remaining_slots -= 1

            if remaining_slots > 0 and pd.notnull(product_type):
                questions_to_store.append({
                    "image_id": row["image_id"],
                    "question": f"What type of product is this?",
                    "answer": product_type.lower(),
                    "options": [product_type.lower(), "shoe", "bag", "shirt"]
                })

            vqa_data.extend(questions_to_store)
            print(f"Stored {len(questions_to_store)} questions for image {image_path}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

        # Follow rate limits (15 RPM = 5 seconds per request)
        time.sleep(5)

    # Save batch
    output_path = f"{output_prefix}_{start_idx}_{end_idx}.csv"
    vqa_df = pd.DataFrame(vqa_data)
    vqa_df.to_csv(output_path, index=False)
    print(f"Saved {len(vqa_df)} VQA questions to {output_path}")
    return vqa_df

# Run for batch
start_idx = 0
end_idx = 10
generate_vqa_batch(start_idx, end_idx)

# Preview
print("\nSample VQA data from batch:")
vqa_df = pd.read_csv(f"{output_prefix}_{start_idx}_{end_idx}.csv")
print(vqa_df.head())