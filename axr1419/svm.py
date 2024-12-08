import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, itr=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.itr = itr
        self.w = None
        self.b = None

    def fit(self, X, y):
        '''
        To Train the Model SVM.
        Arguments:
            X input array of features of shape (n_samples, num_features).
            y input array of labels of shape (n_samples,) belongs to set {-1,1}.
        '''
        _, num_features = X.shape
        # Intializing the weights
        self.w = np.zeros(num_features)
        self.b = 0

        # Converting the y into comptabile format {if y > 1 = then y = 1 otherwise -1}
        for i in range(len(y)):
            if (y[i] <= 0):
                y[i] = -1
            else:
                y[i] = 1

        #Actual Training
        for _ in range(self.itr):
            for index, x_i in enumerate(X):
                z = np.dot(x_i, self.w) + self.b
                line = y[index] * z
                mul = 2 * self.lambda_param * self.w
                diff = np.dot(x_i, y[index])
                if (line >= 1):
                    # Gradient calculation for a correctly classified point
                    self.w -= self.learning_rate * mul
                else:
                    # Gradient calculation for wrongly classified point
                    self.w -= self.learning_rate * (mul - diff)
                    self.b -= self.learning_rate * y[index]

    def predict(self, X):
        """
        This function predict the output.
        Args:
            X input matrix of feature of shape (n_samples, num_features).
        Returns:
            Predicted Array of labels of shape (n_samples,).
        """
        lnr_opt = np.dot(X, self.w) + self.b
        result = np.sign(lnr_opt) 
        return result

class TSVM:
    def __init__(self, base_svm, max_itr=100, lambda_u=0.1):
        """
        Transductive Support Vector Machine.
        Args:
            base_svm: An instance of the SVM class for handling labeled data.
            max_iter: Maximum number of iterations for optimizing the TSVM.
            lambda_u: Weight for the loss term involving unlabeled data.
        """
        self.base_svm = base_svm
        self.max_itr = max_itr
        self.lambda_u = lambda_u

    def fit(self, X_l, y_l, X_u):
        """
        Training SVM model using the X_labeled , y_labeled and X_unlabeled.
        Args:
            X_l Labeled array of Input feature of shape (n_labeled_samples, num_features).
            y_l Array of Labels of Labeled target vector of shape (n_labeled_samples,).
            X_u Array of Unlabeled features of shape (n_unlabeled_samples, n_features).
        """
        # Train the SVM on labeled Data
        self.base_svm.fit(X_l, y_l)
        w, b = self.base_svm.w, self.base_svm.b

        # Assigned the Pseudo labels for unlabeled data
        lnr_opt = np.dot(X_u, w) + b
        pseudo_labels = np.sign(lnr_opt)
        pseudo_labels[pseudo_labels == 0] = 1  # To handle the boundary cases

        for _ in range(self.max_itr):
            # Concatetenate the labeled and pseudo-labeled data
            X_combined = np.concatenate((X_l, X_u), axis=0)
            y_combined = np.concatenate((y_l, pseudo_labels), axis=0)

            # train the base SVM 
            self.base_svm.fit(X_combined, y_combined)

            # Update Weights & Bias
            w, b = self.base_svm.w, self.base_svm.b

            # Update the assigned pseudo-labels
            lnr_opt = np.dot(X_u, w) + b
            new_pseudo_labels = np.sign(lnr_opt)
            new_pseudo_labels[new_pseudo_labels == 0] = 1

            # Checking the convergence
            if np.all(new_pseudo_labels == pseudo_labels):
                break

            pseudo_labels = new_pseudo_labels

        # Store final weights and bias
        self.w, self.b = w, b

    def predict(self, X):
        '''
        This function predict the output.
        Args:
            X input matrix of feature of shape (n_samples, num_features).
        Returns:
            Predicted Array of labels of shape (n_samples,).
        '''
        lnr_opt = np.dot(X, self.w) + self.b
        result = np.sign(lnr_opt)
        return result

    
def train_tsvm(X_labeled, y_labeled, X_unlabeled, max_itr=50, learning_rate=0.01, lambda_param=0.01, lambda_u=0.1, n_iters=100):
    """
    Train a Transductive Support Vector Machine for the labled and unlabled data

    Args:
        X_labeled, y_labeled, X_unlabeled, learning_rate=0.01 have standand meaning
        lambda_param: This is used as a parameter of Regularization
        lambda_u :Weight for the loss term for unlabeled data similar to learning Rate
        n_iters :No of Trining Iterations for Base SVM  and max_itr : max no of iterations for TSVM

    Returns:
        TSVM: An object of trained TSVM model.
    """
    # Create the base SVM model
    base_svm = SVM(learning_rate=learning_rate, lambda_param=lambda_param, itr=n_iters)
    
    # Initialize the TSVM
    tsvm = TSVM(base_svm=base_svm, max_itr=max_itr, lambda_u=lambda_u)
    
    # Train the TSVM
    tsvm.fit(X_labeled, y_labeled, X_unlabeled)
    
    return tsvm

class MulticlassTSVM:
    def __init__(self, n_classes, max_iter=100, learning_rate=0.01, lambda_param=0.01, lambda_u=0.1, n_iters=100):
        """
        Multi-Classifier TSVM using One-vs-Rest strategy.
        Args:
            n_classes : No of classes in the given dataset.
            max_iter ,learning_rate ,lambda_param ,lambda_u, n_iters : have same defination as defined above classess.
        """
        self.n_classes = n_classes

        # Intializing the individual TSVM classifier for each class which give the probability of the membership of that class 
        # classifier having the highest probability, corresponding class will be the redicted class
        self.classifiers = []
        for _ in range(n_classes):
            base_svm = SVM(learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
            tSVM = TSVM(base_svm, max_iter=max_iter, lambda_u=lambda_u)
            self.classifiers.append(tSVM)

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        Training of  multiclassifier TSVM using the Labeled and unlabeled data
        Args:
            X_labeled, y_labeled, X_unlabeled : Have the same defination as we used in above classes.
        """

        for class_idx in range(self.n_classes):
            # Convert class labels into binary labels
            y_binary = np.where(y_labeled == class_idx, 1, -1)
            self.classifiers[class_idx].fit(X_labeled, y_binary, X_unlabeled)

    def predict(self, X):
        """
        Predict class labels for the given data.
        Args:
            X : Input array of Features (n_samples, n_features).
        Returns:
            Array of Predicted labels of shape (n_samples,).
        """
        d_s = []
        for classifier in self.classifiers:
            d_s.extend(classifier.predict(X))
        d_s = np.array(d_s)

        # Finalised the class with the higest decision score
        return np.argmax(d_s, axis=0)
