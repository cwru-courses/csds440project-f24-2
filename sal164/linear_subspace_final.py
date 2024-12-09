import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import t_functions

# Create a shared model for the two views
def create_shared_model(input_dim=784, encoding_dim=64):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(encoding_dim, activation='relu')(x)
    model = Model(inputs, x)
    return model

# Contrastive Loss function
def contrastive_loss(y_true, y_pred, margin=1.0):
    distance = tf.norm(y_pred, axis=-1, keepdims=True)
    loss = y_true * tf.square(distance) + (1 - y_true) * tf.maximum(margin - distance, 0)
    return tf.reduce_mean(loss)

def train_contrastive_model(X1_train, X2_train, y_train, batch_size=32, encoding_dim=64, epochs=10, margin=1.0):
    # Create shared model
    shared_model = create_shared_model(input_dim=X1_train.shape[1], encoding_dim=encoding_dim)
    
    # Define the two views
    view1 = shared_model(X1_train)
    view2 = shared_model(X2_train)
    
    # Create the model for contrastive learning
    model = Model(inputs=[X1_train, X2_train], outputs=[view1, view2])
    
    model.compile(optimizer='adam', loss=contrastive_loss)
    model.fit([X1_train, X2_train], y_train, batch_size=batch_size, epochs=epochs)
    
    return model

# Apply the learned representations
def apply_learned_representations(model, X_train):
    shared_model = model.get_layer('model')
    representations = shared_model.predict(X_train)
    return representations