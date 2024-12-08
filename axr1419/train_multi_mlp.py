import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlp_multi import MLP

# Load the dataset i.e. CIFAR-10
data = load_dataset("uoft-cs/cifar10")

# To convert the image data into arrays use preprocess function
def preprocs_data(imgs):
    flattened_imgs = []
    for img in imgs:
        flattened_imgs.append(np.array(img).flatten())
    flattened_imgs_array = np.array(flattened_imgs)
    return flattened_imgs_array

# Assign data to the corresponding Task variable
X_train = preprocs_data(data['train']['img'])
y_train = np.array(data['train']['label'])
X_test = preprocs_data(data['test']['img'])
y_test = np.array(data['test']['label'])

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# For training purposes y_labeled shoud be in shape (Batch, num_sample) One-hot encode the labels for MLP training
num_cls = len(np.unique(y_train))
y_labeled_one_hot = np.eye(num_cls)[y_train]
mlp = MLP(input_size=X_train.shape[1], n_classes=num_cls)

# Hyper-parameters for training
epochs = 10000
learning_rate = 0.01

for epoch in range(epochs):
    # Forward pass
    outputs = mlp.feed_forward_block(X_train)

    # Compute Cross-Entropy Loss added 1e-7 to handle 0 exception
    mul = y_labeled_one_hot * np.log(outputs + 1e-7)
    sums = np.sum(mul, axis=1)
    loss = -np.mean(sums)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Backward pass for backpropagation to perform weight updation
    mlp.backpropagation(y_labeled_one_hot, learning_rate=learning_rate)

# Make predictions on the test set
y_pred = np.argmax(mlp.feed_forward_block(X_test), axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save weights after training
weights_filepath = "multi_mlp_weights.pkl"
mlp.save_weights(weights_filepath)
