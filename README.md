# Vision Transformer Implementation and Classification with PyTorch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Red)

This repository contains the overall proceduer of implementing Vision Transformer (ViT), classifying 30 Types of Plants Image.

Inspired by the logic of the paper *"An Image is Worth 16x16 Words : Transformers For Image Recognition At Scale"*, this project implements the original structure of the vision transformer designed on paper.


## Datasets

The dataset contains 30 different types of platns images, including 21000 training images, 3000 validation images and 6000 test images, with a total size of 1.48GB, and supports the recognition of the folloiwng plants types : aloevera, banana, bilimbi, cantalope, cassava, coconut, etc.



Download Link : [Kaggle Plants Classification](https://www.kaggle.com/datasets/marquis03/plants-classification)

## Repository Structure

```text
.
├── LinearProjection.py      # Implementatino of Linear Projectino to understand ViT.
├── README.md              
├── main_augmentation.py     # Main Execution File with Mix-up Augmentation Skill.
├── network.py               # Implementation of ViT Architecture in detail.
```

## Resutls
### Top-Record Model : Plain-ViT with Mix-up Augmentation

**Model with Mix-up augmentation recorded 0.739 validation accuracy and the other models w/o augmentation recorded about 0.52 and 0.65 for each.**


<img width="452" height="140" alt="image" src="https://github.com/user-attachments/assets/3ded9e0e-2b96-4704-ba0d-1a6cde7314d2" />


As we analyzed the ratio of correctly classified and misclassified images
Model with augmentation confused Melon with Cantaloupe

As we can see below, the original dataset from kaggle contains the same image on different types of class folder. 


<img width="372" height="240" alt="image" src="https://github.com/user-attachments/assets/8fe836ed-3e12-4db8-9f29-8b8641d9022c" />

## Visualization
### Patch Embedding

## Reference
* Paper : *"An Image Worth 16x16 Words : Transformers for Image Recognition At Scale"*
