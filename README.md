# 🌱PlantNet39 - Plant Leaf Disease Classification (39 Classes)

PlantNet39 is a deep learning based **Computer Vision system for plant leaf disease classification**, capable of identifying **39 different crop disease and healthy classes** from leaf images. The project focuses on **practical deployment** , balancing accuracy , model size , and inference efficiency. 

👉 **Live demo (Hugging Face Spaces):** https://huggingface.co/spaces/roshan454/PlantNet39

## Problem Statment 

Plant disease significantly impact agricultural yield and food security.
Manual Inspection is:
- Time consuming
- Prone to errors
- Expensive

This project explores whether a **lightweight deep learning model** can reliably classify plant leaf disease and be **deployed for real-time use.** 

## Model & Architecture 
- **Backbone:** EfficientNet-B2 (pretrained on ImageNet)
- **Approach:** Transfer Learning
- **Trainable parameters:** ~55k
- **Frozen parameters:** ~7.7M

EfficientNet-B2 was selected after evaluating CNN and ViT-based approaches, 
mainly due to:
- Lightweight (EfficientNet-B2 is **~35-36 MB** and ViT B-16 is around **~330-350 MB.** )
- Deployability on CPU-only environements
- Performance almost near to ViT

## Dataset 
**Classes:** 39 
   - 38 plant leaf classes (disease + healthy)
   - 1 background/non-leaf class (`Background_without_leaves`) used to detect inputs that are not plant leaves
**Data Split:** Train/Test(folder-structured)
**Image Size:** 224-288px(EffNet Native Transforms)

> Dataset is downloaded programmatically during training and not stored directly in the repository.

## Data Pipeline 
- Image resizing & normalization using EfficientNet weights
- TrivialAugmentWide for data augmentation
- Batched Dataloaders for efficient training

## Training Details 
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Optimizer:** Adam (lr = 1e-3)
- **Epochs:** 10
- **Device:** GPU (training), CPU (deployment)
> Label smoothing was used to improve generalization and reduce over-confidence on visually similar disease classes.

## Results 
- The model learns strong visual patterns across multiple crops
- Performs well on common disease classes
- Occasional Confusion between visually similar diseases

> The goal of the project is deployability rather than high-end accuracy.

## Deployment 

The trained model is deployed using **Gradio** and hosted on **Hugging Face Spaces.**

**Features:**
- Upload a leaf image
- Recieve class probabilities for all 39 classes
- CPU-only inference
- Fast response time (~milliseconds per image)
- The demo perfectly handles invalid inputs by predicting a dedicated non-leaf background class.
  
## Limitations
- Performance depends on image quality and lighting
- Confusion between visually similar diseases
- Dataset Bias affect rare disease classes

## Future Improvements 
- Grad-CAM visual explanations
- Fine-grained class balancing
- Mobile-friendly deployment
- Multimodal inputs (image + metadata)
- Lightweight vision transformer comparison

## Tech Stack 
- Python
- PyTorch
- Torchvision
- EfficientNet
- Gradio
- Hugging Face Spaces

## 👨‍💻 Author
**Roshan Mohammed**
<br>MSc Artificial Intelligence
<br>Focused on applied ML, computer vision, and deployable deep learning systems.


