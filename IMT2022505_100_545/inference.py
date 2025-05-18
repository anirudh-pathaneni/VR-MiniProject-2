import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers.models.auto.processing_auto  import AutoProcessor
from transformers import BlipForQuestionAnswering
from peft import PeftModel
from transformers.utils.quantization_config import BitsAndBytesConfig
from huggingface_hub import login
import re

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run VQA inference with fine-tuned BLIP-VQA-Base model")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Authenticate with Hugging Face
    hf_token = "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" 
    login(hf_token)
    print("Hugging Face login successful")

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Verify required columns
    required_columns = ['image_name', 'question']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

# Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check bf16 support (for CUDA)
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    model_dtype = torch.bfloat16 if is_bf16_supported else torch.float32
    print(f"Using model dtype: {model_dtype}")

    # Quantization config (for CUDA only)
    quantization_config = None
    if device.type == "cuda":
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=model_dtype,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8"
            )
        except Exception as e:
            print(f"Failed to initialize 8-bit quantization: {e}. Falling back to non-quantized mode.")
            quantization_config = None

# Load processor
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base", token=hf_token)
    base_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        torch_dtype=model_dtype,
        quantization_config=quantization_config
    )
    model = PeftModel.from_pretrained(base_model, "5unnySunny/blip-vqa-base", token=hf_token, torch_dtype=model_dtype)
    print("LoRA model loaded successfully")

    model = model.to(device)
    model.eval()

    # Set pad_token_id
    # Ensure tokenizer has pad_token
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing VQA"):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = f"{row['question']}. Return a single word."

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                images=image,
                text=question,
                return_tensors="pt",
                padding="max_length",
                max_length=16,
                truncation=True
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=5,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40
                )
            answer = processor.decode(outputs[0], skip_special_tokens=True).strip().lower()
            answer = re.sub(r"[^\w\s]|'s|\?s", "", answer)
            answer = answer.split()[0] if answer.split() else "unknown"
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            answer = "error"

        generated_answers.append(answer)

    # Save results
    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    main()