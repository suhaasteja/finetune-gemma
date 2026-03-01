# Project: PII Detection in Images with PaliGemma

## 🎯 Overview
This project establishes a complete pipeline for fine-tuning a Vision-Language Model (VLM) to detect and localize Personally Identifiable Information (PII) within image-based documents. Due to the sensitive nature of real-world PII, this pipeline utilizes **Synthetic Data Generation (SDG)** using a combination of high-fidelity background textures and LLM-generated content.

---

## 🏗️ Technical Architecture

### 1. Vision-Language Model (VLM)
- **Model**: `google/paligemma-3b-pt-224`
- **Architecture**: Combines a SigLIP vision encoder with a Gemma language model.
- **Fine-tuning Technique**: **LoRA (Low-Rank Adaptation)** for parameter-efficient training, allowing for fine-tuning on consumer-grade GPUs (e.g., NVIDIA T4/A10).
- **Format**: Uses a "Detect" prompt (`detect [classes]`) returning normalized bounding boxes in the format `<locY1><locX1><locY2><locX2>`.

### 2. Synthetic Data Content Generation
- **Legacy Option**: `google/gemma-2-2b-it` (Local execution)
- **Active Option**: **Google Gemini API** (`gemini-1.5-flash`)
- **Purpose**: Generates diverse, ethnically varied, and contextually accurate PII data (Names, SSNs, Passport numbers) in structured JSON format.
- **Benefits**: Faster generation, higher quality, and zero local hardware requirements for the generation phase.
- **Diversity**: Overrides standard `Faker` limitations by using LLM reasoning to generate data specific to document types (e.g., international passport formats vs. US driver's licenses).

### 3. Image Synthesis Engine
- **Backgrounds**: High-resolution (1024px) realistic textures including:
    - Scanned office paper grain.
    - Distressed/crumpled aged paper.
    - Professional medical/registration form layouts.
- **Augmentation**: Random placement, text-ghosting/shadow effects for realism, and automatic coordinate-to-PaliGemma-token conversion.

---

## 📂 Project Structure & Files

| File | Description |
| :--- | :--- |
| `pii_detection_finetune.ipynb` | Main pipeline: Model loading, LoRA config, and Training Loop. |
| `synthetic_data_generation.ipynb` | Image merging logic: Overlays PII onto 4 custom template types (Drivers, Insurance, Passport, SSN). |
| `gemma_pii_content_gen.ipynb` | Content engine: Queries Gemma 2 2B to create 80+ unique PII JSON profiles. |
| `PROJECT_SUMMARY.md` | Technical documentation and project overview. |

---

## 🏷️ Target PII Classes
The model is trained to recognize 5 specific entity types:
1. `person_name`: Full legal names.
2. `ssn`: US Social Security Numbers.
3. `credit_card`: Full and partial credit/debit card numbers.
4. `dob`: Dates of birth in various formats.
5. `national_id`: Passport numbers, Driver's License numbers, and Member IDs.

---

## 🛠️ Required Dependencies
```bash
pip install transformers peft bitsandbytes accelerate datasets faker pillow
```

## 🚀 Future Roadmap
- [ ] Implement **OCR-Grounded Pre-training** for improved text accuracy.
- [ ] Add **Blur/Occlusion Augmentation** to handle low-quality photos.
- [ ] Export to **TFLite/ONNX** for mobile/edge deployment.
