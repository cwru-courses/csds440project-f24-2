# Group N Comparison

## Methods

The methods in this study all come from our original seed paper. Overall, it discusses the motivations and assumptions behind semi-supervised learning. Semi-supervised learning is motivated because labeled data is often expensive and time-consuming to obtain, while unlabeled data is abundant and readily available. Many methods assume that the data lies in a low-dimensional space, and that points close to each other in that space are likely to have the same label, or that the decision boundary between classes should lie in low-density regions. The paper goes over the most important semi-supervised learning algorithms.

1. **Method 1**: Harmonic Minimization Method: The harmonic minizmization method is a graph based learning algorithm for semi-supervised learning. It constructs a graph where each data point is a node, and the edges represent the similarity between them. The algorithm minimizes a cost function that encourages similar points to have similar labels while respecting the known labels, iteratively updating the labels of the unlabeled data points by solivng a harmonic function until convergence.  This method was contributed by Weston Mansier. [Weston's Report](wlm35/Weston_Mansier_Project_Report.md).

2. **Method 2**: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. This method was contributed by [Prasanna Kumar peram]. You can find more details in their individual report: [https://github.com/cwru-courses/csds440project-f24-2/blob/main/pxp488/PrasannaKumarPeram_Project_Report.md].
3. **Method 3**: Brief description of Method 3. This method was contributed by [Group Member 3 Name]. You can find more details in their individual report: [link to individual report 3].
1. **Method 1**: Brief description of Method 1. This method was contributed by Stephanie. You can find more details in their individual report: [link to individual report 1].
2. **Method 2**: Brief description of Method 2. This method was contributed by [Group Member 2 Name]. You can find more details in their individual report: [link to individual report 2].
3. **Two-view Feature Generation Model**: 

The key idea in this approach is to define a set of auxiliary problems to enhance semi-supervised learning. Unlike co-training, where each view is assumed to be sufficient for classification, this method uses one view ($z_2$) to predict some function of the other view ($m(z_1)$), with $m$ indexing different auxiliary problems. These auxiliary tasks can be trained on unlabeled data.

A linear model, $w_m^\top z_2$, is used to fit $m(z_1)$, and the weight vector $w_m$ is learned from all the unlabeled data. Each weight vector $w_m$ has the same dimension as $z_2$. Some dimensions of the set of weights may be more important, indicating that corresponding dimensions in $z_2$ are more useful for the task.

Singular Value Decomposition (SVD) is applied to the matrix of weights to extract a compact representation of $z_2$, and a similar transformation is done for $z_1$ by swapping the roles of $z_1$ and $z_2$. The original representation $(z_1, z_2)$ is then concatenated with the new representations of $z_1$ and $z_2$ to create a new feature vector. This new representation incorporates information from unlabeled data and the auxiliary problems, and is used in standard supervised learning with labeled data. The selection of appropriate auxiliary problems is very important to this approach.

## Results, Analysis, and Discussion

### Experiments

Experiment: Analyzing the impact of varying unlabeled set sizes on model performance
### Experiment

- **Dataset**: For this experiment, we used the MNIST image classification dataset. 
- **Methodology**: Semi-supervised learning algorithms separate the dataset into partitions where some examples are given labels and other examples are stripped of theirs. The proportion of examples that do not have labels has a significant impact on the performance of the method. Ideally, the methods will perform well with very few available labels. For this experiment, we set the unlabeled sets to be of proportions 0.9, 0.95, and 0.98 of the entire dataset. We then plotted their accuracy to get a measure of which methods perform best when very little labeled data is provided.

### Results

- **Results for Method 1**:
![Weston's results](wlm35_group_plot.png)
**MNIST dataset**

- **Results for Method 1**: Provide the outcome of experiments conducted using Method 1.
- **Results for Method 2**:
       Different times different epoch values are given to check the validation accuracy and traning loss
          Below is a table summarizing the average accuracy and loss over different epochs during the training process:(mean- teachers)
          
          | Epochs | Average  Accuracy (%) | Average Training Loss |
          |--------|----------------------|-----------------------|
          | 10     | 97.61                | 0.08237               |
          | 20     | 98.43                | 0.0355                |
          | 30     | 98.40                | 0.0280                |
          
          This table illustrates the progression of the model's performance, showing improvements in accuracy and reductions in loss as the number of training epochs increases.
          
          
          Below is a table summarizing the average accuracy and loss over different epochs during the training process: (for mean- teachers+mixmatch)
          
          | Number of Epochs | Average Training Loss | Average  Accuracy (%) |
          |------------------|-----------------------|---------------------------------|
          | 10               | 1.73                  | 87.37                           |
          | 20               | 1.69                  | 87.53                           |
          | 30               | 1.63                  | 88.78                           |



- **Two-view Feature Generation Model**:
![ahh](synthetic_varying_proportion.png)

This model did not perform better than a base logistic regression model.


### Analysis and Discussion

- **Analysis of Method 1**: The results of this experiment on the MNIST dataset are similar to the results provided in the initial paper, with slightly lower performance after the research extension of dynamic weights was applied.

- **Best Performing Methods**: Which methods performed the best overall? Describe why these methods performed well in the given context.
- **Research Extensions**: Discuss how extensions or variations of the methods worked. Which adaptations were successful, and why?
- **Key Insights**: Summarize what the group learned from comparing these methods. Were there any surprises or unexpected findings? What challenges did the group face during this comparison?
- **Conclusions**: Provide the final conclusions drawn from the comparison. Based on the results and analysis, what do you recommend for future experiments or improvements?

