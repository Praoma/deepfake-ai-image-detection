# Deepfake & AI-Generated Image Detection Using Deep Learning

A deep-learning project for classifying images as **AI-generated or human-created**, featuring a comparative evaluation of EfficientNet-B0, ResNet-50, and ConvNeXt-Tiny, alongside image-quality robustness testing and Grad-CAM interpretability.

> **Project status:** Experimental prototype. The evaluation covers the ADS504 dataset and selected image-quality transformations. Generalization to genuinely unseen image generators has not yet been established.

## Overview

AI-generated images are becoming increasingly realistic, making it challenging to distinguish synthetic images from human-created content. This project investigates how different pretrained deep-learning architectures perform on this binary classification task.

The project includes:

- Comparative evaluation of three deep-learning architectures.
- Transfer learning and fine-tuning of pretrained models.
- Held-out test-set evaluation using multiple classification metrics.
- Robustness experiments involving JPEG compression and image downscaling.
- Error analysis using a confusion matrix and classification report.
- Grad-CAM visualizations to inspect regions associated with model predictions.

**Best test accuracy:** 85.87% — ConvNeXt-Tiny.

## Models Evaluated

| Model | Architecture family | Test accuracy |
|---|---|---:|
| EfficientNet-B0 | Efficient convolutional network | 82.47% |
| ResNet-50 | Residual convolutional network | 84.93% |
| ConvNeXt-Tiny | Modern convolutional network | **85.87%** |

The models use pretrained ImageNet weights. Their classification heads were adapted for two classes, followed by classifier training and fine-tuning of selected backbone layers.

## Dataset

**Dataset:** [ADS504-Image-Arrays](https://huggingface.co/datasets/tkbarb10/ADS504-Image-Arrays)

| Split | Number of images |
|---|---:|
| Training | 11,998 |
| Validation | 1,499 |
| Test | 1,500 |
| **Total** | **14,997** |

- Input resolution: 224 × 224 pixels
- Image format: RGB
- Task: Binary image classification

### Class mapping

| Label | Class |
|---:|---|
| 0 | AI-generated |
| 1 | Human-created |

The dataset was resized to a common resolution. Although its source collections include multiple image collections, per-image generator provenance was unavailable in the loaded dataset. Consequently, the test split should not be treated as proof of generalization to unseen generators.

## Methodology

1. **Dataset inspection:** Examined split sizes, class balance, image dimensions, and dataset metadata.
2. **Preprocessing:** Resized images to 224 × 224, converted them to RGB, and applied ImageNet normalization.
3. **Data augmentation:** Applied random horizontal flipping during training.
4. **Transfer learning:** Initialized each architecture with pretrained ImageNet weights and replaced its classification head.
5. **Fine-tuning:** Trained the classification head and subsequently fine-tuned selected later layers.
6. **Model selection:** Used validation loss to select checkpoints.
7. **Final evaluation:** Compared the selected models on the held-out test split.
8. **Robustness testing:** Evaluated ConvNeXt-Tiny under JPEG compression and downscaling.
9. **Interpretability:** Inspected misclassified examples and generated Grad-CAM heatmaps.

The notebook uses a random seed of 42 and a batch size of 32.

## Experimental Results

### 1. Held-out test performance

| Model | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 82.47% | 82.49% | 82.46% | 91.04% |
| ResNet-50 | 84.93% | 84.95% | 84.93% | **93.23%** |
| ConvNeXt-Tiny | **85.87%** | **85.85%** | **85.85%** | 93.17% |

**Key findings**

- ConvNeXt-Tiny achieved the highest test accuracy, balanced accuracy, and macro F1.
- ResNet-50 achieved the highest ROC-AUC, narrowly ahead of ConvNeXt-Tiny.
- EfficientNet-B0 performed below the other two models in this experiment.
- The small ROC-AUC difference between ResNet-50 and ConvNeXt-Tiny should not be interpreted as a statistically significant advantage.

### 2. Image-quality robustness

ConvNeXt-Tiny was evaluated on the original test images and transformed versions of those images.

| Test condition | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|
| Original images | 85.87% | 85.85% | 93.17% |
| JPEG quality 75 | 80.60% | 80.55% | 89.12% |
| JPEG quality 50 | 78.53% | 78.48% | 87.51% |
| Downscale to 56 × 56, then resize | 69.80% | 69.65% | 81.23% |

**Interpretation:** Performance declined under the tested image-quality transformations, with the largest decline occurring after downscaling.

These experiments demonstrate sensitivity to the specific transformations evaluated. They do not establish performance under every real-world image degradation.

### 3. Error analysis

ConvNeXt-Tiny produced the following results on the 1,500-image test set:

- Correctly classified AI-generated images: 669
- AI-generated images classified as human-created: 86
- Human-created images classified as AI-generated: 126
- Correctly classified human-created images: 619
- Total errors: 212
- Overall error rate: 14.13%

The inspected mistakes included portraits, artwork, everyday objects, and other scenes. These examples motivate further investigation, but they do not establish the underlying causes of the model's errors.

### 4. Grad-CAM interpretability

Grad-CAM was applied to selected correctly classified and misclassified images using the final feature stage of ConvNeXt-Tiny.

The visualizations showed variation in highlighted regions. In some examples, heatmaps extended beyond the primary subject, suggesting that contextual image regions may influence predictions.

Grad-CAM is a qualitative interpretability method. Its heatmaps do not prove causal reasoning or provide definitive explanations for individual predictions.

## Limitations

This project has several important limitations:

1. **Unseen-generator generalization is unverified.** An independent, provenance-verified external dataset has not yet been used for evaluation.
2. **Dataset bias may affect results.** Source-collection characteristics and potential source overlap may influence classification performance.
3. **Robustness testing is limited.** Only the listed JPEG compression and downscaling transformations were evaluated.
4. **Interpretability is approximate.** Grad-CAM visualizations cannot establish the exact reasons for model predictions.
5. **Statistical uncertainty was not quantified.** Confidence intervals and statistical significance tests were not calculated.

## Future Improvements

- Evaluate on an independent dataset containing images from verified unseen generators.
- Investigate cross-dataset generalization and source-specific biases.
- Test additional image transformations and resolutions.
- Evaluate confidence calibration and decision thresholds.
- Develop a lightweight web application for image uploads and predictions.
- Expand interpretability and systematic error analysis.

## Repository Structure

The intended repository structure is:

```text
deepfake-ai-image-detection/
├── README.md
├── DeepFake_Detection_V2.ipynb
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py
│   └── predict.py
└── results/
    ├── test_metrics.csv
    ├── robustness_metrics.csv
    ├── experiment_summary.json
    └── gradcam_examples/
```

The source files and result artifacts should be added as they are organized. The structure above describes the planned layout, not files already verified to exist in the repository.

## Reproducibility

The experiments were conducted in Google Colab using GPU acceleration and PyTorch-based models.

The project artifacts include trained checkpoints for EfficientNet-B0, ResNet-50, and ConvNeXt-Tiny, along with consolidated test and robustness metrics.

For reproducibility, preserve the experiment configuration, package versions, preprocessing pipeline, and selected checkpoints alongside the notebook.

## Disclaimer

This project is an experimental image classifier, not a definitive forensic detector. Its predictions should not be treated as proof that an image is AI-generated or human-created. Further independent evaluation is required before making claims about generalization to unseen generators or real-world forensic reliability.
