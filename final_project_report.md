# Generalizable AI-Generated Image Detection

## 1. Objective
Develop and compare deep-learning classifiers for distinguishing
AI-generated images from human-created images, with particular
attention to model comparison and image-quality robustness.

## 2. Dataset
Dataset: tkbarb10/ADS504-Image-Arrays
- Training images: 11,998
- Validation images: 1,499
- Test images: 1,500
- Image size: 224 x 224
- Class 0: AI-generated
- Class 1: Human-created

The dataset includes images associated with multiple source collections,
but per-image generator provenance is unavailable in the loaded dataset.
Consequently, the split does not establish generalization to unseen generators.

## 3. Models and training
The following pretrained architectures were evaluated:
- EfficientNet-B0
- ResNet-50
- ConvNeXt-Tiny

The experiment used a shared preprocessing pipeline, ImageNet
normalization, a batch size of 32, and a fixed random seed of 42.
Training included classifier-head training and fine-tuning selected
later backbone layers. Validation loss was used for checkpoint selection.

## 4. Held-out test results

| Model | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 82.47% | 82.49% | 82.46% | 91.04% |
| ResNet-50 | 84.93% | 84.95% | 84.93% | 93.23% |
| ConvNeXt-Tiny | 85.87% | 85.85% | 85.85% | 93.17% |

ConvNeXt-Tiny achieved the highest test accuracy and macro F1.
ResNet-50 achieved a marginally higher ROC-AUC. The small AUC difference
should not be interpreted as evidence of a meaningful statistical advantage.

## 5. Image-quality robustness
The ConvNeXt-Tiny model was evaluated on the original test images and
versions transformed using JPEG compression or downscaling.

| Condition | Accuracy | ROC-AUC |
|---|---:|---:|
| Original | 85.87% | 93.17% |
| JPEG quality 75 | 80.60% | 89.12% |
| JPEG quality 50 | 78.53% | 87.51% |
| Downscale to 56x56, then resize | 69.80% | 81.23% |

Performance declined under the tested image-quality transformations,
especially downscaling. These results indicate sensitivity to the
specific transformations evaluated; they do not establish performance
under every real-world image degradation.

## 6. Error analysis
The held-out test evaluation produced 212 errors among 1,500 images,
corresponding to a 14.13% error rate for ConvNeXt-Tiny.

Confusion matrix:
- AI-generated correctly classified: 669
- AI-generated classified as human-created: 86
- Human-created classified as AI-generated: 126
- Human-created correctly classified: 619

The sample of visually inspected mistakes included portraits, artwork,
objects, and other scenes. These examples motivate further investigation
but do not establish the cause of model errors.

## 7. Interpretability
Grad-CAM was applied to selected correctly classified and misclassified
examples using the final ConvNeXt feature stage. Heatmaps varied across
images and sometimes highlighted regions outside the main subject.
These visualizations are qualitative: they do not prove causal reasoning
or identify a definitive explanation for each prediction.

## 8. Limitations
1. Generalization to genuinely unseen image generators has not been
   validated using an independent, provenance-verified external dataset.
2. Dataset source overlap and collection biases may influence performance.
3. Only the specified JPEG and downscaling transformations were tested.
4. Grad-CAM provides approximate visual explanations.
5. No confidence intervals or statistical significance tests were performed.

## 9. Conclusion
ConvNeXt-Tiny was the strongest model by test accuracy and macro F1 on
ADS504, while ResNet-50 achieved a slightly higher ROC-AUC. Image-quality
experiments revealed substantial performance degradation under compression
and especially downscaling. The results support performance on this dataset
and under the tested transformations, but do not yet demonstrate
generalization to unseen generators.

## 10. Reproducibility
Model checkpoints, test metrics, robustness metrics, and experiment
configuration are included in the accompanying project package.
