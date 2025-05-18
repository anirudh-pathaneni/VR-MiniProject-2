# Visual Question Answering (VQA) Mini-Project Report

## Description
This project focuses on creating and fine-tuning a Visual Question Answering (VQA) system using the Amazon Berkeley Objects (ABO) dataset. We leveraged the Gemini 2.0 API to curate a high-quality dataset with single-word answer questions, aligned with image content. Two pre-trained models from Salesforce, `Salesforce/blip-vqa-base` and `Salesforce/blip2-flan-t5-xl`, were evaluated and fine-tuned using Low-Rank Adaptation (LoRA) on Kaggle GPUs to adapt to our curated dataset. The project aims to optimize VQA performance for concise, visually inferable answers.

## Project Structure
The project files are hosted in the GitHub repository at [https://github.com/anirudh-pathaneni/VR-MiniProject-2](https://github.com/anirudh-pathaneni/VR-MiniProject-2). The structure is as follows:
- `IMT2022505_100_545/`: Directory containing inference scripts, added in the latest commit.
- `Blip-2-Base.ipynb`: Kaggle notebook with base model evaluation and metrics for `Salesforce/blip2-flan-t5-xl`.
- `Blip-vqa-base-Base.ipynb`: Kaggle notebook with base model evaluation and metrics for `Salesforce/blip-vqa-base`.
- `LoRA-Blip-vqa-base-Seq.ipynb`: Kaggle notebook with LoRA fine-tuning and metrics for `Salesforce/blip-vqa-base`.
- `README.md`
- `blip-vqa-base_finetune_1.csv`: CSV file with fine-tuning metrics for `Salesforce/blip-vqa-base`.
- `blip-vqa-base_metrics_base.csv`: CSV file with base metrics for `Salesforce/blip-vqa-base`.
- `blip2-base_metrics_base.csv`: CSV file with base metrics for `Salesforce/blip2-flan-t5-xl`.
- `dc.py` and `dp.py`: Python scripts for data curation code and QA dataset generation.
- `vqa_dataset_final_new.csv`: Final curated QA dataset in CSV format.

The CSV files contain performance metrics, while the Jupyter notebooks (`.ipynb`) are Kaggle notebooks used for model training, evaluation, and experimentation.

## Metrics
The models were evaluated using the following metrics on the curated VQA dataset:
- **Accuracy**: Proportion of correct predictions.
- **F1 Score**: Harmonic mean of precision and recall, reflecting performance with class imbalance.
- **BERTScore**: Semantic similarity between predicted and ground truth answers using BERT embeddings.

| Model                       | Accuracy  | F1 Score  | BERTScore |
|-----------------------------|-----------|-----------|-----------|
| `Salesforce/blip-vqa-base`  | 0.323236  | 0.088085  | 0.96276   |
| `Salesforce/blip2-flan-t5-xl` | 0.267772 | 0.09119   | 0.94183   |
| LoRA-`Salesforce/blip-vqa-base` | 0.67382 | 0.27207 | 0.98236   |
| LoRA-`Salesforce/blip2-flan-t5-xl` | ---     | ---      | ---       |

Note: Metrics for LoRA-`Salesforce/blip2-flan-t5-xl` are unavailable due to memory constraints.

## Observations
- The KV Cache was successfully utilized, enhancing inference efficiency for both models.
- XFormers attention provided better throughput than default PyTorch attention, though Flash Attention was incompatible with our setup.
- Quantization with 8-bit precision helped manage Kaggle's memory limits, enabling faster training for `Salesforce/blip-vqa-base`.
- Fine-tuning `Salesforce/blip-vqa-base` with LoRA showed significant improvements, with accuracy rising from 32% to 67%.
- Fine-tuning `Salesforce/blip2-flan-t5-xl` encountered an `OutOfMemoryError: CUDA out of memory` despite reducing `per_device_train_batch_size` and `per_device_eval_batch_size`, and setting `lora_dropout=0.1`. Training was limited to 1 epoch, indicating the need for further optimization strategies.
- `Salesforce/blip-vqa-base` outperformed `Salesforce/blip2-flan-t5-xl` in accuracy and BERTScore, likely due to its specialized VQA training, while `Salesforce/blip2-flan-t5-xl`’s broader design struggled with fine-grained visual recognition.