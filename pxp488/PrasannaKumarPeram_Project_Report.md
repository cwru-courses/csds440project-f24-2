## 1. Survey

**paper 1** Xie, Qizhe, et al. "Self-training with noisy student improves imagenet classication." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

in this papaer The main method is based on the concept of knowledge distillation, where a larger, more complex teacher model trains a smaller student model. However, the twist in this approach is the introduction of noise to the student model during training, referred to as "Noisy Student Training."
**Key Ideas and Techniques:**
- The student model is trained to mimic the output distribution of the teacher model, which has already been trained on labeled data.
- to make the student model more better and generalizing random data augmentation techniques are applied, and noise is introduced in the form of dropout and stochastic depth.
- after student model trained it becomes teacher and cycle is repeated. with each iteration the model is usally expeted to perform better.

This method implemented of this self-training method resulted in significant improvements over the baseline model on the ImageNet classification challenge. also helps in model generalizatoin on unseen data.

**Strengths:**
1. The introduction of noise gives the student model to learn more generalizable features patterns rather than memorizing the training data.
2. it works very well on inlabelled data.
3. The approach disscussed increases the accuracy compared to traditional models which perfomed imagent dataset.

**Weaknesses:**
1. The iterative process of refining the student and teacher models can be computationally expensive, if dataset is large especially.
2. The quality of the initial teacher model heavily influences the performance gains, which potentially limit the approach if indial model is not that very strong.


**paper 2** Miyato, Takeru, et al. "Virtual adversarial training: a regularization method for supervised and semi-supervised learning." IEEE transactions on pattern analysis and machine intelligence 41.8 (2018): 1979-1993

in this paper by  virtual adversarial loss a technique designed to improve the robustness and generalization of machine learning model. works for both supervised and semi-supervised  learnings  also emphasizes model stability in response to local perturbations around input data points.

**Key Ideas and Techniques:**
1. VAT involves generating small, adversarially crafted perturbations to the input data that are likely to fool the model.
2. by adding these perturbations in the training process the VAT effectively serves as a regularization technique, helping to prevent overfitting by smoothing the model's output distribution as discussed in the paper.
3. increases the generalization and stability of models.

**Strengths:**
1. we can say enhaced generalizayion because improved perfomance on unseen data.
2.  VAT is effective in both supervised and semi-supervised learning environments.
3. it reduces the sensituvity to small input changes also enhancing the model's capabilities against adversarial attacks.

**weakness** 
1. Generating virtual adversarial examples and recalculating the loss can be computationally intensive, especially for large datasets.
2. The effectiveness of VAT can depend heavily on the choice of hyperparameters, such as the perturbation size.

**paper 3** Sohn, Kihyuk, et al. "Fixmatch: Simplifying semi-supervised learning with consistency and condence." arXiv preprint arXiv:2001.07685 (2020).

in this paper  a simplified approach to semi-supervised learning that builds on the ideas of consistency regularization and pseudo-labeling is proposed. FixMatch assumes that predictions should be consistent across different augmentations of the same image and uses high-confidence predictions from the model as labels for training on unlabeled data.

**Key Ideas and Techniques:**
1. consistency regularization which means  Ensures that the model's predictions are consistent across various augmentations of the same input.
2. confidence thresholding means Only predictions with confidence above a certain threshold are used for training, reducing the risk of reinforcing incorrect predictions.(confidence inerval prediction as dicussed in the class).

**Strengths:** 
1. by combining consistency and confidence in a straightforward manner this method reduces the complexity of semi-supervised learning pipelines.
2. Less computationally demanding than methods that require multiple forward passes or complex data manipulations.
3. Makes efficient use of available unlabeled data by focusing on high-confidence predictions.

**Weaknesses:**
1.  confidence threshold,  may not always accurately reflect the correctness of predictions the methods efficiency can be limited this.
2. The performance depends on the effectiveness of the chosen data augmentations.


## 2. Method

The paper implimented by me is **"Mean teachers are better role models:Weight-averaged consistency targets improve semi-supervised deep learning results"** by Tarvainen and Valpola. This method leverages the consistency of the model outputs over different training epochs by averaging the model weights, rather than the outputs themselves.

what is **Weight Averaging**?
The core of the Mean Teacher model is the weight averaging mechanism, where the teacher model's weights are updated as an exponential moving average (EMA) of the student model's weights over time. Mathematically it can be described using the 
            $$
            \theta'_t = \alpha \theta'_{t-1} + (1 - \alpha) \theta_t
            $$

Here....
- $\theta'_t$ is the teacher model weights at training step $t$.
- $\theta_t$ is the student model weights at training step $t$.
- $\alpha$ is the decay factor which controls the rate at which the teacher model's weights are updated which is towards the student's weights.

It ensures that the teacher model remains stable and the model accumulates the knowledge gained from various states of the student model across the different training iterations.

One more key concept is **Consistency Cost** .
Consistency Cost minimizes the difference in outputs between the student and teacher models for the same input, under different perturbations. This is implemented is done by a consistency loss function, here we are using the squared error (MSE) as the consistency cost. 
The consistency cost defined follows...
                    $$
                    \text{Consistency Cost} = \frac{1}{N} \sum_{i=1}^{N} (y_i^{\text{teacher}} - y_i^{\text{student}})^2
                    $$

Here....
- $y_i^{\text{teacher}}$ and $y_i^{\text{student}}$ are the outputs of the teacher and student models, respectively, where is the same input $x_i$.
- $N$ is the count total samples in the batch.

By learning from the teacher's more reliable forecasts, this loss motivates the student model to generate predictions that are similar to the teacher model's, enhancing the student model's capacity for generalization.


How **Training** Works?
The training procedure goes as follows,  alternating between updating the student model using the standard backpropagation method based on a supervised loss when labeled data is available and  combined with the consistency cost, and updating the teacher model using the weight averaging formula. The student model is trained on mixture of labeled and unlabeled data here labelled data is used for supervised loss and where as unlabelled data contributes to the consistency cost.

**Implimentation**
1. We initialize the student and teacher models separately , while initilization the teacher model starts with same as student model weights.
2. To feed into the student and teacher models we generate perturbed versions of the input data for each batch of the data.
3. Then we calculate the supervised loss using labelled data for the student model and the consistency cost by using the both labelled and unlabelled   data.
4. **Update the Model** :- we update the model using the gradients from the all combined losses. we update the teacher model's weights usinf the above mentined EMA formula 


## 3. "Research" section

In meanteacher data augmentation is utilized to enforce consistency across multiple views of the same data, helping to stabilize the learning process and improve the robustness and accuracy of the model.  

$$\| p_{\text{student}}(y | \text{Augment}(x); \theta_{\text{student}}) - p_{\text{teacher}}(y | \text{Augment}(x); \theta_{\text{teacher}}) \|^2$$

**Role of augumentation in mean-teacher**
1. for every input multiple augumented versions are generated using transformation like rotation, cropping, adding, noise etc. it is done for labelled and unlabelled data.
2. both student model and teacher model make predictions on these augumented version of same input data.  this is very imporatant because it allows consistency across the different views of same data.
3. primary traning objective in mean teacher is consistency loss which is measuring the difference between the student and teacher models on the augumented data as shown in above equation. the goal is to minize the loss.
4. diverse range of data augumentation to expose the model for different range of scenarios. this diversity ensure that model not just memorizing the training data but actually learn different patterns of the data.
5. The student model predicts outputs for x1 while the teacher model operates with lagged parameters updated via EMA approach predicts output x2.
6. in mean teahcer the data is augumented two times . which is **weak** augumentation i think . Thts why i am considering the approach **mixmatch** .

**Mixmatch approach**
 This method is mainly designed to unify the labelled and unlabelled data under a single framework by gussing the low entropy labels for unlabelled data.
 Also mixing labelled and unlabelled data using the mixup,a technique that forms linear interpolations of pairs of examples and their labels.
 1. for every image in both labelled and unlabelled dataset , mix match first apllies the stochastic data augumentation , generally cropping, flipping , rotation etc. this augumentation helps in creating the diverse representation of the same image which helps to learn different patterns of the data.
 2. here for each unlabelled data the mix match applied the augumentation multiple times (more that 2). This results in several different versions of the same image, each potentially highlighting different features or aspects of the image.
 3. each version of unlabelled image is passed through model to predic the probabilities for class. The average of these predictions across multiple augmented versions forms a sharp (low-entropy) pseudo-label. the thing here is that averaging across the multiple predictions tends to increase the conficence of the predictions, thereby making the pseudo-labels more reliable.
 4. This averaged prediction may undergo "sharpening" . the sharpening involves adjusting the temperature of the softmax output to make the distribution more peaked means enchance the most probable class probability and diminish others.
 5. MixUp Regularization :- mixmatch then uses the mixup augumentation not just the between the pair of images or pairs of unlabelled images but also between the pairs of labelled and unlabelled images. This is crucial because it creates synthetic examples that are combinations of both labeled and unlabeled data, effectively teaching the model to generalize across the spectrum of provided data.

 **befits of this augumentation**
- Training in different augumented data gives different learing patterns of the data which increases the Robustness and Invariance.
- Different rage of augumented data can reduce the chaces of models overfitting.
- The use of muliple augumentations generates more reliable pseudo labels for unlabelled data which is used for training purpose.
- To imporve the confidence and accuracy of predictions MixUp creates examples that lie at the boundary of class distributions, teaching the model to handle ambiguous cases better and smoothing the decision boundaries .


**MixMatch and Mean Teacher Integration**

**Data Augmentation**:- both labelled and un-labelled data goes into extensive data augumentation which generates mulple version of the same input which is important for enforcing consistency and leveraging the Mean Teacher model.

**Batch Processing**
Each training batch is composed of a mix of labeled and unlabeled data, enhancing the model's ability to learn from both explicit labels and different patterns of the data.

**Consistency Regularization with Mean Teacher**
the student model predicts labels for different versions of augumented data, consistency losss is calculted whic is to reduce the discrepancy
student and teacher model predictions .The teacher model’s parameters are updated as the exponential moving average (EMA) of the student model’s parameters, providing a stable target for the student model's training.

The is **euquation** can:- 
$$\| p_{\text{student}}(y | \text{Augment}(x); \theta_{\text{student}}) - p_{\text{teacher}}(y | \text{Augment}(x); \theta_{\text{teacher}}) \|^2$$

The above equation ensures consistency across student and teacher predictions, enhancing prediction reliability and reducing overfitting.

**Entropy Minimization**
- Sharpen the model’s predictions on unlabeled data, thereby making the outputs more definitive.
-Minimize the entropy of the predictions for unlabeled data to encourage the student model to output more confident and less entropic predictions.
The is **euquation** can:- 
\[
H(p, p_{\text{student}}(y | x; \theta_{\text{student}})) = -\sum_{i} p_i \log(p_i)
\]

The above equation reduces the entropy helps improve the clarity of the model’s outputs, which is very important for training stability and accuracy.

**Pseudo-Labeling**

High-confidence predictions from the student model on unlabeled data are converted into pseudo-labels, which are then used as training targets within the cross-entropy loss, simulating labeled data.


**Algorithm Steps with Mean Teacher**

1.  Apply stochastic data augmentation to both labeled and unlabeled samples.
2. The student model predicts labels for all of the augmented versions of each data point from the learning data. Concurrently, teacher model  preditcts labels for same data using the smothed parameters
3. Implement MixUp regularization by combining pairs of examples (labeled with unlabeled) to create virtual training examples, using both student and teacher outputs to guide the combination.
4. Fincally we we calculate the loss that integrates consistency regularization to update the model. Simultaneously, update the teacher model parameters as the EMA of the student’s parameters.
   

Below is a detailed breakdown of the loss function used in MixMatch:

**Components of the Loss Function**
- **\( L_C \)**: Cross-entropy loss (labeled examples)
- **\( L_U \)**: Unsupervised loss for unlabeled examples, focusing on minimizing the discrepancy between pseudo-labels and model predictions.
- **\( \lambda \)**: A balancing factor that weights the contribution of the unsupervised loss relative to the supervised loss.

**Mathematically defined as**
The overall loss function combines these components as follows:
$$
L = L_C + \lambda L_U
$$


- **Cross-Entropy Loss (\( L_C \))**:
  $$ 
  L_C = -\sum_{(x, p) \in \mathcal{X}^L} H(p, P_{model}(y | x; \theta))
  $$
  where, \( H \) denotes the entropy function, \( p \) are the true labels, and \( P_{model}(y | x; \theta) \) are the probabilities predicted by the model for labeled data \( x \).

- **Unsupervised Loss (\( L_U \))**:
  $$
  L_U = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \|q - P_{model}(y | u; \theta)\|^2
  $$
  where \( q \) represents the pseudo-labels generated from the model's predictions on augmented versions of the unlabeled data \( u \), and \( P_{model}(y | u; \theta) \) is the prediction after applying a sharpening function to make the pseudo-labels more distinct.

**MixUp Augmentation**
Additionally, MixMatch employs MixUp augmentation to further regularize the model:
$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$
$$
\tilde{p} = \lambda p_i + (1 - \lambda) p_j
$$

## Results, Analysis and Discussion**

- on MNIST dataset


Below is a table summarizing the average accuracy and loss over different epochs during the training process:(mean- teachers)

| Epochs | Average  Accuracy (%) | Average Training Loss |
|--------|----------------------|-----------------------|
| 10     | 97.61                | 0.08237               |
| 20     | 98.43                | 0.0355                |
| 30     | 98.40                | 0.0280                |

This table illustrates the progression of the model's performance, showing improvements in accuracy and reductions in loss as the number of training epochs increases.


Below is a table summarizing the average accuracy and loss over different epochs during the training process: (for mean- teachers_mixmatch)

| Number of Epochs | Average Training Loss | Average  Accuracy (%) |
|------------------|-----------------------|---------------------------------|
| 10               | 1.73                  | 87.37                           |
| 20               | 1.69                  | 87.53                           |
| 30               | 1.63                  | 88.78                           |

The even though the perfomance is increased with number of epochs compared to mean teaches the accuracy is very less and the traning loss is also increased for mixmatch added approach. 

- The mean teachers method works beter compared to mean-teachers+mixmatch.
- The resons  
    1. mix match introduces the complexity into traning processs which includes data augumentation and mixing strategies which might not did well on this dataset characterstics or rask.
    2. MixMatch and Mean Teachers together involve a complex interaction of parameters like the Beta distribution parameters in MixUp, weight decay rates, etc. 
    3. the MNIST data characterstics might be more amenable to straightforward consistency techniques rather than more complex mixing and augmentation strategies. Cetrains datasets with specific kind of noise or vulnarability or might not benfit as for additional complexity techniques.
    4. The additional constraints and regularizations imposed by MixMatch might lead the model to underfit, particularly if the pseudo-labels generated are not accurate or the interpolated samples do not represent meaningful inputs for the model.

## Conclusions
- The research extension did not work as expected compared to mean teachers  because of above stated reasons. we have used the unlabelled data by giving pseudo label but it did not work properly. 
**futurewor** :- Check the different approaches stated in the survey also check the results in other datasets also. may be explore different augumentation techiniques which will work this this kind of specifi datasets. 



**Note** i have used the chatGPT to give me the markdown code for the euqations and tables .