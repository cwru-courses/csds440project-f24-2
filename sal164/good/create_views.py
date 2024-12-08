from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
#from keras.layers import Input, Dense, Lambda
#from keras.models import Model
#import tensorflow as tf

def create_views_feature_split(X_unlabeled, split_ratio=0.5):
    split_point = int(X_unlabeled.shape[1] * split_ratio)
    X1_unlabeled = X_unlabeled[:, :split_point]
    X2_unlabeled = X_unlabeled[:, split_point:]
    return X1_unlabeled, X2_unlabeled

'''
def create_views_autoencoder(X_unlabeled, encoding_dim=64):
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


'''def create_views_vae(X_unlabeled, encoding_dim=64):
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
    return X1_unlabeled, X2_unlabeled'''


def create_views_pca(X, n_components=784):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)  # Perform PCA transformation
    
    # Split into odd and even indexed principal components
    X1 = X_pca[:, ::2]  # View 1: Take all odd indexed components (0, 2, 4, ...)
    X2 = X_pca[:, 1::2]  # View 2: Take all even indexed components (1, 3, 5, ...)
    
    return X1, X2