import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from mlp_multi import MLP  # Ensure MLP is implemented correctly
from svm import MulticlassTSVM  # Assuming this is defined correctly

# Load CIFAR-10 dataset
data = load_dataset("uoft-cs/cifar10")

# Preprocess CIFAR-10 images and labels
def preprocess_images(images):
    return np.array([np.array(img).flatten() for img in images])

X_train = preprocess_images(data['train']['img'])
y_train = np.array(data['train']['label'])
X_test = preprocess_images(data['test']['img'])
y_test = np.array(data['test']['label'])

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split into labeled and unlabeled data for semi-supervised learning
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# One-hot encode the labels for MLP training
n_classes = len(np.unique(y_train))
y_labeled_one_hot = np.eye(n_classes)[y_labeled]

# Initialize the MLP model
mlp = MLP(input_size=X_labeled.shape[1], n_classes=n_classes)

# Load pre-trained weights
weights_file = "multi_mlp_weights.pkl"
mlp.load_weights(weights_file)

# Train the MLP model if fine-tuning is required
n_epochs = 10  # Fine-tuning epochs (optional)
learning_rate = 0.01

print("Fine Tuning the MLP")
for epoch in range(n_epochs):
    outputs = mlp.feed_forward_block(X_labeled)

    # Compute Cross-Entropy Loss
    loss = -np.mean(np.sum(y_labeled_one_hot * np.log(outputs + 1e-7), axis=1))
    mlp.backpropagation(y_labeled_one_hot, learning_rate=learning_rate)

# Extract features from labeled, unlabeled, and test data
X_labeled_features = mlp.extract_features(X_labeled)
X_unlabeled_features = mlp.extract_features(X_unlabeled)
X_test_features = mlp.extract_features(X_test)

# Train and test the TSVM
multi_tsvm = MulticlassTSVM(
    n_classes=n_classes,
    max_iter=50,
    learning_rate=0.01,
    lambda_param=0.01,
    lambda_u=0.1,
    itr=100
)

print("Training TSVM...")
multi_tsvm.fit(X_labeled_features, y_labeled, X_unlabeled_features)

print("Testing TSVM...")
y_pred = multi_tsvm.predict(X_test_features)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
