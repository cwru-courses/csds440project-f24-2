# Group N Comparison

## Methods

Brief description of the methods compared in this study and attribute each method to the respective group member.

1. **Method 1**: Harmonic Minimization Method: The harmonic minizmization method is a graph based learning algorithm for semi-supervised learning. It constructs a graph where each data point is a node, and the edges represent the similarity between them. The algorithm minimizes a cost function that encourages similar points to have similar labels while respecting the known labels, iteratively updating the labels of the unlabeled data points by solivng a harmonic function until convergence.  This method was contributed by Weston Mansier. [Weston's Report](wlm35/Weston_Mansier_Project_Report.md).

2. **Method 2**: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. This method was contributed by [Prasanna Kumar peram]. You can find more details in their individual report: [https://github.com/cwru-courses/csds440project-f24-2/blob/main/pxp488/PrasannaKumarPeram_Project_Report.md].
3. **Method 3**: Brief description of Method 3. This method was contributed by [Group Member 3 Name]. You can find more details in their individual report: [link to individual report 3].

## Results, Analysis, and Discussion

### Experiments

Experiment: Analyzing the impact of varying unlabeled set sizes on model performance

- **Dataset**: For this experiment, we used the MNIST image classification dataset. 
- **Methodology**: Semi-supervised learning algorithms separate the dataset into partitions where some examples are given labels and other examples are stripped of theirs. The proportion of examples that do not have labels has a significant impact on the performance of the method. Ideally, the methods will perform well with very few available labels. For this experiment, we set the unlabeled sets to be of proportions 0.9, 0.95, and 0.98 of the entire dataset. We then plotted their accuracy to get a measure of which methods perform best when very little labeled data is provided.

### Results

- **Results for Method 1**:
![Weston's results](wlm35_group_plot.png)
- **Results for Method 2**: Provide the outcome of experiments conducted using Method 2.
- **Results for Method 3**: Provide the outcome of experiments conducted using Method 3.

### Analysis and Discussion

- **Analysis of Method 1**: The results of this experiment on the MNIST dataset are similar to the results provided in the initial paper, with slightly lower performance after the research extension of dynamic weights was applied.

- **Best Performing Methods**: Which methods performed the best overall? Describe why these methods performed well in the given context.
- **Research Extensions**: Discuss how extensions or variations of the methods worked. Which adaptations were successful, and why?
- **Key Insights**: Summarize what the group learned from comparing these methods. Were there any surprises or unexpected findings? What challenges did the group face during this comparison?
- **Conclusions**: Provide the final conclusions drawn from the comparison. Based on the results and analysis, what do you recommend for future experiments or improvements?

