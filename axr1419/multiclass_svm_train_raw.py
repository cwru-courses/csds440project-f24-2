import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mlp_multi import MLP  
from svm import MulticlassTSVM  

# Load the dataset CIFAR-10
data = load_dataset("uoft-cs/cifar10")

# Flatened the images into Arrray for traing and testing
def preprocess_img(imgs):
    flattened_imgs = []
    for img in imgs:
        flattened_imgs.append(np.array(img).flatten())
    flattened_imgs_array = np.array(flattened_imgs)
    return flattened_imgs_array

# Intializing the trainging and testing data to corresponding Variable
X_train = preprocess_img(data['train']['img'])
y_train = np.array(data['train']['label'])
X_test = preprocess_img(data['test']['img'])
y_test = np.array(data['test']['label'])

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the data into labled and unlabled set for training and Testing
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# One-hot encode the labels for MLP training
n_classes = len(np.unique(y_train))
y_labeled_one_hot = np.eye(n_classes)[y_labeled]

# Intialising the parameters of TSVM for multi-classification
multi_tsvm = MulticlassTSVM(
    n_classes=n_classes,
    max_iter=50,
    learning_rate=0.01,
    lambda_param=0.01,
    lambda_u=0.1,
    itr=100
)

print("Training TSVM...")
multi_tsvm.fit(X_labeled, y_labeled, X_unlabeled)

print("Testing TSVM...")
y_pred = multi_tsvm.predict(X_test)

# Evaluate accuracy of the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

precision = precision_score(y_test, y_pred, average='macro')  # Options: 'macro', 'micro', 'weighted', 'binary'
print(f"Precision: {precision:.2f}")

# Recall
recall = recall_score(y_test, y_test, average='macro')
print(f"Recall: {recall:.2f}")

# F1-Score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-Score: {f1:.2f}")
