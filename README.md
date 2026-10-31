# Deep Learning — Atelier 2
## Dataset and environment

MNIST contains 60,000 training images and 10,000 test images
of handwritten digits from 0 to 9.

The experiments used PyTorch in Google Colab with an NVIDIA T4 GPU.

## Models

- **CNN:** two convolutional layers with ReLU and max pooling,
  followed by fully connected layers and dropout.
- **Faster R-CNN:** a small custom backbone, with digit bounding
  boxes generated from nonzero pixels. Labels 1–10 represent
  digits 0–9; label 0 is reserved for background.
- **VGG16:** ImageNet pretrained weights, with the last
  convolutional block and final classification layer trained.
- **AlexNet:** ImageNet pretrained weights, with frozen
  convolutional features and a trainable classifier.
- **ViT:** built from scratch using 49 patches, 8-dimensional
  embeddings, a classification token, fixed positional encoding,
  two transformer blocks and two attention heads.

CNN, Faster R-CNN, VGG16 and AlexNet were trained for 3 epochs.
ViT was trained for 5 epochs.

## Results

| Model | Test accuracy | Macro F1 | Test classification loss | Training time (s) |
|---|---:|---:|---:|---:|
| CNN | 98.65% | 0.9864 | 0.0390 | 29.41 |
| Faster R-CNN | 99.29% | 0.9929 | Not measured | 1509.91 |
| VGG16 | 99.21% | 0.9920 | 0.0236 | 1018.59 |
| AlexNet | 99.20% | 0.9919 | 0.0233 | 356.95 |
| ViT | 87.23% | 0.8694 | 0.3969 | 649.45 |

Faster R-CNN achieved the highest digit recognition accuracy,
while the CNN offered the best balance between accuracy and
training time in these runs. Bounding-box quality was not evaluated.

AlexNet achieved almost the same accuracy as VGG16 while taking
about 35% of its training time.

The small ViT obtained lower training and test accuracies,
suggesting underfitting with this configuration and training budget.
This does not establish that ViTs generally perform worse than CNNs.

The experiments used different preprocessing, pretraining and
training budgets, so the results compare these particular setups.
Detection loss is not directly comparable to classification loss.

## Running the notebook

Open `lab2.ipynb` in Google Colab, select a GPU runtime,
and execute the cells from top to bottom. MNIST and the
pretrained weights are downloaded automatically.

Dependencies: torch, torchvision, scikit-learn, pandas,
matplotlib and tqdm.

## ViT reference and adaptations

The ViT section follows Brian Pulfer's tutorial:
[Vision Transformers from Scratch (PyTorch): A step-by-step guide](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c).

The implementation processes attention across the batch,
stores fixed positional encoding as a buffer, and returns
raw classification scores for CrossEntropyLoss.

## What I learned

I practiced GPU training, CNN construction, transfer learning,
and adapting a classification dataset for object detection.
Building the ViT helped me understand patch embeddings,
positional encoding, attention and residual connections.
Comparing the models showed why training cost matters
alongside prediction accuracy.
