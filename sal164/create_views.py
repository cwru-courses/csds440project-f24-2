from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import numpy as np

from keras.layers import Input, Dense, Lambda
from keras.models import Model
import tensorflow as tf

def create_views_feature_split(X_unlabeled, split_ratio=0.5):
    split_point = int(X_unlabeled.shape[1] * split_ratio)
    X1_unlabeled = X_unlabeled[:, :split_point]
    X2_unlabeled = X_unlabeled[:, split_point:]
    return X1_unlabeled, X2_unlabeled


'''def create_views_autoencoder(X_unlabeled, encoding_dim=64):
    scaler = StandardScaler()
    X_unlabeled_scaled = scaler.fit_transform(X_unlabeled)
    input_layer = Input(shape=(X_unlabeled.shape[1],))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(X_unlabeled.shape[1], activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_unlabeled_scaled, X_unlabeled_scaled, epochs=20, batch_size=256, shuffle=True)
    X1_unlabeled = encoder.predict(X_unlabeled_scaled)
    X2_unlabeled = X_unlabeled_scaled
    return X1_unlabeled, X2_unlabeled'''


def create_views_kernel_pca(X_unlabeled, kernel='rbf', n_components=64):
    scaler = StandardScaler()
    X_unlabeled_scaled = scaler.fit_transform(X_unlabeled)
    kpca = KernelPCA(kernel=kernel, n_components=n_components)
    X_unlabeled_transformed = kpca.fit_transform(X_unlabeled_scaled)
    X1_unlabeled = X_unlabeled_scaled
    X2_unlabeled = X_unlabeled_transformed
    return X1_unlabeled, X2_unlabeled


def create_views_vae(X_unlabeled, encoding_dim=64):
    scaler = StandardScaler()
    X_unlabeled_scaled = scaler.fit_transform(X_unlabeled)
    inputs = Input(shape=(X_unlabeled.shape[1],))
    z_mean = Dense(encoding_dim)(inputs)
    z_log_var = Dense(encoding_dim)(inputs)
    z = Lambda(lambda x: x[0] + tf.exp(0.5 * x[1]) * tf.random.normal(shape=(encoding_dim,)))([z_mean, z_log_var])
    encoder = Model(inputs, z_mean)
    decoder_h = Dense(encoding_dim, activation='relu')
    decoder_mean = Dense(X_unlabeled.shape[1], activation='sigmoid')
    z_decoded = decoder_mean(decoder_h(z))
    vae = Model(inputs, z_decoded)
    xent_loss = X_unlabeled.shape[1] * tf.keras.losses.binary_crossentropy(inputs, z_decoded)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae_loss = xent_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.fit(X_unlabeled_scaled, epochs=20, batch_size=256)
    X1_unlabeled = encoder.predict(X_unlabeled_scaled)
    X2_unlabeled = X_unlabeled_scaled
    return X1_unlabeled, X2_unlabeled


def create_views_pca(X, n_components=784):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # Perform PCA transformation
    
    # Split into odd and even indexed principal components
    X1 = X_pca[:, ::2]  # View 1: Take all odd indexed components (0, 2, 4, ...)
    X2 = X_pca[:, 1::2]  # View 2: Take all even indexed components (1, 3, 5, ...)
    
    return X1, X2

def rotate_image(image, angle_deg=90):
    """
    Rotate an image by a given angle in degrees using matrix operations and bilinear interpolation.
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Get the image shape (height, width)
    height, width = 28, 28

    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Get the center of the image
    center = np.array([height / 2, width / 2])

    # Create a meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(height), np.arange(width))
    
    # Flatten the grid for easier computation
    coords = np.vstack([x.ravel(), y.ravel()])

    # Translate the coordinates to the center, apply the rotation, and then translate back
    new_coords = np.dot(rotation_matrix, coords - center[:, None]) + center[:, None]

    # Round the new coordinates to integers (this will introduce some interpolation)
    new_coords = np.round(new_coords).astype(int)

    # Initialize an empty image for the rotated version
    rotated_image = np.zeros_like(image)

    # Clip the coordinates to ensure they stay within the bounds of the image
    new_coords[0] = np.clip(new_coords[0], 0, height - 1)
    new_coords[1] = np.clip(new_coords[1], 0, width - 1)

    # Assign pixel values to the new coordinates in the rotated image
    rotated_image[new_coords[0], new_coords[1]] = image[x.ravel(), y.ravel()]

    return rotated_image

def create_views_rotation(X):
    """
    Generate two views for each image:
    - View 1: Original image
    - View 2: Rotated image (by 90 degrees)
    """
    X1 = X  # View 1: Original image
    
    # Rotate each image in X by 45 degrees using matrix operations
    X2 = np.array([rotate_image(image, 90) for image in X])  # View 2: Rotated images
    
    return X1, X2

def create_views_flip(X):
    """
    Generate two views for each image:
    - View 1: Original image
    - View 2: Flipped image (horizontally)
    """
    X1 = X  # View 1: Original image
    X2 = np.array([np.fliplr(image) for image in X])  # View 2: Flipped images
    
    return X1, X2

def add_noise(image, amount=0.05):
    """
    Add salt-and-pepper noise to an image.
    """
    noisy_image = image.copy()
    total_pixels = noisy_image.size
    num_noise_pixels = int(total_pixels * amount)
    
    # Add salt noise (set random pixels to 1)
    salt_coords = [np.random.randint(0, i-1, num_noise_pixels) for i in noisy_image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 1
    
    # Add pepper noise (set random pixels to 0)
    pepper_coords = [np.random.randint(0, i-1, num_noise_pixels) for i in noisy_image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return image, noisy_image

