import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Create a shared model for the two views
def create_shared_model(input_dim=784, encoding_dim=64):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(encoding_dim, activation='relu')(x)
    model = Model(inputs, x)
    return model

# Contrastive Loss function
def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss function:
    - y_true: Binary labels (1 for same class, 0 for different classes).
    - y_pred: Euclidean distance between the two view representations.
    """
    distance = tf.norm(y_pred, axis=-1, keepdims=True)
    loss = y_true * tf.square(distance) + (1 - y_true) * tf.maximum(margin - distance, 0)
    return tf.reduce_mean(loss)

# Train the contrastive model
def train_contrastive_model(X1_train, X2_train, y_train, batch_size=32, encoding_dim=64, epochs=10, margin=1.0):
    # Create shared model
    shared_model = create_shared_model(input_dim=X1_train.shape[1], encoding_dim=encoding_dim)
    
    # Create the views
    input1 = layers.Input(shape=(X1_train.shape[1],))
    input2 = layers.Input(shape=(X2_train.shape[1],))
    view1 = shared_model(input1)
    view2 = shared_model(input2)

    # Create the contrastive learning model
    model = Model(inputs=[input1, input2], outputs=[view1, view2])
    
    model.compile(optimizer='adam', loss=contrastive_loss)

    # Train the model
    model.fit([X1_train, X2_train], y_train, batch_size=batch_size, epochs=epochs)
    
    return model

# Apply the learned representations
def apply_learned_representations(model, X_train):
    # Extract the shared model layer
    shared_model = model.get_layer(index=2)  # Getting the shared model from the trained model
    representations = shared_model.predict(X_train)
    return representations
