# Semi-Supervised Learning with TSVM and MLP

This project explores the combination of **Transductive Support Vector Machines (TSVM)** and **Multi-Layer Perceptron (MLP)** for semi-supervised learning. The aim is to assess the impact of feature extraction via MLP on the performance of TSVM in a semi-supervised setting, particularly using the **CIFAR-10 dataset**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Results](#results)
4. [Research and References](#research-and-references)
5. [License](#license)

## Project Overview

In this project, we combine **MLP (Multi-Layer Perceptron)** and **TSVM (Transductive Support Vector Machines)** to create a hybrid model. The key idea is to use MLP as a feature extractor for TSVM. By passing the output of the second hidden layer of the MLP to the TSVM, we investigate how feature extraction influences the classification performance. Additionally, TSVM is also trained directly on the raw data to compare the performance between the two approaches.

### Key Components:
- **MLP (Feature Extractor)**: A neural network with two hidden layers.
- **TSVM (Transductive Support Vector Machines)**: A semi-supervised model that uses both labeled and unlabeled data to create a decision boundary with maximum margin.
- **CIFAR-10 Dataset**: A standard benchmark dataset for image classification tasks.

## Setup and Installation

To set up the environment and run the code, follow these steps:

### Prerequisites
All the Libraries are installed in the requirements.txt file.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/tsvm-mlp-semi-supervised.git
    cd tsvm-mlp-semi-supervised
    ```

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

1. **Train MLP**:
    - The MLP model is trained on the CIFAR-10 dataset and saved for feature extraction.
    - To train the MLP, run:
      ```bash
      python train_multi_mlp.py
      ```

    - I used MLP as a feature extractor so its a pre trained Model. Just Download the weights file from here.
    - [Download the weights file for Pre-Trained MLP](https://drive.google.com/file/d/1rCT68lC38pNhwkjeblTUQ0MmAyxYGs7W/view?usp=sharing)
      

2. **Train TSVM**:
    - After training the MLP, the output from the second layer of MLP is passed to TSVM for semi-supervised learning.
    - To train TSVM on the output of MLP, run:
      ```bash
      python multiclass_svm_train.py
      ```

    - To train TSVM directly on the raw CIFAR-10 data, run:
      ```bash
      python multiclass_svm_train_raw.py
      ```

3. **Evaluate Results**:
    - After training, the model's accuracy, precision, recall, and F1-score are calculated and printed.
    - To evaluate the model, run:
    - Refer to the Analysis.ipynb 

## Results

### TSVM without MLP (raw data):
- **Accuracy**: 0.1000
- **Precision**: 0.01
- **Recall**: 1.00
- **F1-Score**: 0.02

### TSVM with MLP:
- **Accuracy**: 0.1000
- **Precision**: 0.01
- **Recall**: 1.00
- **F1-Score**: 0.02

Both models showed similar performance in this particular setup. Further optimization of the MLP model or TSVM parameters could improve results.

## Research and References

### Summary of Relevant Research:
- **Vapnik (1998)**: Introduced the foundation of SVMs and TSVMs, focusing on maximizing margins in classification tasks. TSVMs extend the SVM to include unlabeled data for decision boundary improvement.
- **Bennett & Demiriz (1999), Demirez & Bennett (2000)**: Early TSVM algorithms, highlighting their limitations in scalability and efficiency.
- **De Bie & Cristianini (2004, 2006)**: Proposed semi-definite programming (SDP) approaches to improve TSVM scalability.
- **Chapelle & Zien (2005)**: Introduced gradient-based optimization for TSVM using Gaussian functions to improve training speed.
- **Weston et al. (2006)**: Introduced the concept of a **universum**, where unlabeled data not belonging to either of the two classes helps guide the decision boundary.

For more detailed information, please refer to the references section of the project documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

