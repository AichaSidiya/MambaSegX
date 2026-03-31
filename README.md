# MambaSegX
# Abstract
According to the World Health Organisation (WHO), breast cancer caused 670000 deaths globally in 2022, and it is the primary cause of cancer death in women. Mammography is the primary screening tool, originally analyzed manually. However, with the advent of Computer-Aided-Diagnosis (CAD) systems and deep learning models, we saw architecture like Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) being used for mammography analysis. However, these models face a critical trade-off between computational efficiency and the ability to model long-range dependencies essential for accurate mammogram analysis. Recently, Mamba architecture, based upon a selective State Space Model (SSM), offered a promising alternative with linear computational complexity and effective long-range modeling. However, existing Mamba models designed for medical applications lacked integrated lesion segmentation for precise localization and provided minimal or no explainability, hindering clinical trust and adoption. This research proposes MambaSegX, a mamba-based classification model, with segmentation head for inherent explainability, to consecutively classify breast cancer mammograms then identify lesions. The methodology will involve building 4 models with increasing complexity in-order to achieve higher accuracy. All models will be based on a Mamba encoder for feature extraction and a binary classification (benign vs malignant) decoder. The first model will be the basic Mamba model, then a hybrid model will be built with pretrained Resnet weights, regularization will later be added to reduce overfitting, lastly, DenseNet weights will be added to the original model for higher accuracy, since DenseNet was trained on medical images. After training the classification model, it’s weights will be frozen and a segmentation head will be added to the Mamba backbone for each of the 4 models, the segmentation head will be built using pytorch segmentation library, that provides Unet architecture for semantic segmentation. Later on, Grad-CAM will be used to provide transparent visual justifications for the classification predictions. The model will be trained on the Mini-DDSM dataset, and BreastMNIST will be used to generalize the model by testing the trained models on it. The 4 MambaSegX will be compared and the best one in terms of classification accuracy, segmentation precision (Dice Score), and computational efficiency (FLOPs) will be compared to state-of-the-art models like MedMamba and Mammo-Mamba. The result of this model will bridge the gap between high-performance models and clinical usability by being highly interpretable. 

# Getting DDSM Dataset

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("cheddad/miniddsm2")

print("Path to dataset files:", path)
```python


