# Predicting Wildfire Clusters with Machine Learning

Group 11: Amir, Ryan, Sid, Yuma

### Video


### Introduction/Background

Wildfires are a major threat to our environment, destroying natural habitats and ravaging communities. In 2019, over 4.4 million acres were destroyed by wildfires, and 8 million acres have already been destroyed in 2020 alone [1]. Effects of climate change have resulted in greater wildfire activity over the past couple of years [2], and their ability to rapidly spread out of control puts great importance on extinguishing them as soon as possible. Rapidly detecting and predicting wildfires would allow first responders to act early and extinguish the fires, limiting their disastrous results.

### Problem definition

The goal of this project is to build a predictive model to understand, analyze, and detect patterns in data, allowing us to forecast the locations of wildfires as soon as possible. We have access to a dataset of 1.88 million wildfires in the United States [3], providing 24 years of geo-referenced wildfire records from 1992 to 2015. We are interested in exploring this dataset and applying machine learning techniques to identify patterns or clusters related to identifying causes of wildfires, predicting sizes of wildfires, or identifying wildfire "hotspots"(areas which are more prone to wildfires). The goal is to use machine learning techniques to aid in a real-world application of proper allocation of firefighter resources.


### Data Collection

Specifically, our data is divided into two independent datasets that we will apply feature engineering and data preprocessing independently on, before choosing the most relevant features for our downstream target classification task. The notebooks labeled explore_kaggle and Wildfire Prediction Notebook under the datasets/kaggle and datasets/uci respectively are where you can find all of the code for preprocessing.

#### [UCI Dataset](https://github.com/amirgamil/4641-Project/tree/master/datasets/uci)

The first UCI dataset consists of 517 entries with 13 features describing wildfire instances in a northeast Portuguese national park. These features include month, day, temperature, wind conditions, rain, etc. We perform several important steps on the data to preprocess it:

1. Convert year/month dates into categorical features
2. Scale the columns which contain numerical data (besides the dates which we leave as categorical data) - we use Sci-kit learn's Standard Scaler to create Z-scores for each of our features

##### Before Preprocessing

<img src="report%20materials/uci_before_preprocessing.png">

##### After Preprocessing

![UCI Dataset](report%20materials/uci_after_preprocessing.png)

#### [Kaggle Dataset](https://github.com/amirgamil/4641-Project/tree/master/datasets/kaggle)

The second dataset is a Kaggle dataset consisting of data from over 1.88 million wildfires in the United States. The dataset is initially stored in an SQLite database which we dump into Pandas in order to preprocess it easily. We perform 4 important steps on the dataset to preprocess it:

1. Convert latitude and longitude to standarized forms
2. Convert dates into timestamps with durations that can be processed by our models
3. Scale the columns which contain numerical data - we use Sci-kit learn's Standard Scaler to create Z-scores for each of our features
4. Convert string columns into categorical features

##### Before Preprocessing

![Kaggle Dataset](report%20materials/kaggle_befre_preprocessing.png)

##### After Preprocessing

![Kaggle Dataset](report%20materials/kaggledataset.png)


### Methods

Although previous attempts at wildfire predicted has useed a variety of data, such as meteorological conditions [4], biomass [5], and satellite images [6], there remain challenges in predicting wildfires accurately and understanding the complex patterns by which the environment will respond. [7].
  
Existing research in wildfire spread prediction incorporates high-dimensional features in classification. Tonini et al. use the slopes of land, vegetation type, non-flammable area, etc. to obtain the probability of fire in an Italian region via Random Forest with 15 years of fire damage maps were used as training data [8]. Final predictions ranged from 83.4% to 91.7% accuracy year-to-year under the test dataset. Rodrigues and Riva utilize features such as forest area, power line presence, protected area status, etc. to predict low/high risk in Spanish regions [9]. 30 years of wildfire data and used many regression methods - Random Forest, Boosted Regression Tree, Support Vector Machine, and logistic regression - to predict fire risk. The Random Forest algorithm proves promising with an AUC value of 0.746 vs 0.730, 0.709, and 0.686 for the other algorithms respectively. Sayad et al. use NASA satellite remote sensing data based on crop states, meteorological conditions, thermal intensity, etc. in conjunction with a Multi-layer perceptron neural network and a separate SVM model to predict fire occurrence to an accuracy of 98% and 97% respectively on a small dataset of Canadian fires [10].

Drawing on methods above, we divide our task into two subtasks: unsupervised learning to understand our data and supervised learning to predict the risk of wildfires. 

Our unsupervised learning consisting of building correlation matrices to understand the relationship between features and applying PCA to reduce our feature set to the most important linearly independent features. 

### UCI Unsupervised Learning Results

#### Correlation Matrix

We started by building a correlation matrix, which depict the factors' correlations with each other via a gradient
![Correlation Matrix Results](report%20materials/uci_covariance.PNG)

#### PCA Results
PCA with two components
![PCA Results](report%20materials/uci_PCA.PNG)
Our results from PCA show that there is a lot of noise in our data. None of the classes were linearly inseparable, meaning that none of the features in the UCI dataset alone were strong predictors of our class labels.

### Kaggle Unsupervised Learning Results

#### Correlation Matrix
![Correlation Matrix Results](report%20materials/kaggle_covariance.PNG)

#### PCA Results

PCA with two components.
![PCA Results](/report%20materials/kaggle_PCA.PNG)


The PCA of the Kaggle data set depicts two principal components and the casual factors as the labels. Altough there seems to be a trace of clustering, they are still not clear enough to the point where the model could be used to predict the casual factors. Indeed, the two components are only able to capture 36% of the variance. In order to capture 95% of the variance, eight components were necessary. This makes sense given our data cleaning process because the final dataset contained 9 features which were relatively distinct from each other. 


We also performed a principal component analysis with 3 components and found that the retianed variance was higher but still only 53%.
![3D Kaggle PCA Results: Cause](report%20materials/3D_vis.png)

Additionally, we performed PCA but used States as the target variable,
which presented these graphs:

![Kaggle PCA Results: State](report%20materials/kaggle_state_PCA.png)

![3D Kaggle PCA Results: State](report%20materials/kaggle_state_PCA3D.png)

With larger samples taken:

![3D Kaggle PCA Results: State with larger sample](report%20materials/kaggle_state_PCA3D_more.png)

### Results/Discussion
##### UCI Dataset
We attempted to classify our test wildfire data by fire size code (A - G). In our dataset containing Portuguese National Park fires, we observed a **52.9%** classification accuracy using a Random Forest classification model, **49.0%** accuracy with an AdaBoosted Random Forest classifier, **31.7%** accuracy with a Complement Naive Bayes classifier, **28.8%** accuracy with a Gaussian Naive Bayes classifier, **51.9%** accuracy with a Support Vector Machine classifier, and a **49.0%** accuracy with a Neural Network.

##### UCI Dataset Models
![UCI Dataset](report%20materials/supervised_results_uci.png)

##### UCI Neural Network Results
![UCI Dataset](report%20materials/nn_training_uci.png)

##### UCI Dataset Discussion
A potential reason for our inconclusive classification results may lie in the size of the Portuguese National Park dataset. The dataset only contains 517 instances to analyze. It does, however, have a wide range of features to analyze. The bottleneck in our analysis may be the lack of sufficent amounts of data to launch a large-scale classifier on. Given that our three highest performing classifiers (Random Forest, SVM, and Neural Network) were within 2-3% difference in accuracy, this indicates that our approach to classifying the data or the data itself presented accuracy problems as opposed to our choice of models.

##### Kaggle Dataset Models
We attempted to classify our test wildfire data by fire cause and size. In our dataset containing United States wildfires, we observed a **50.3%** accuracy for classifying the cause of the fire using a Random Forest classification model and **47.0%** accuracy with a Neural Network. For predicting the size of the wildfire, we achieved an accuracy of **4%** using a Random Forest and **5%** accuracy with a Neural Network

##### Kaggle Dataset Results
![Kaggle Dataset](report%20materials/kaggle_results_graph.PNG)

##### Kaggle Dataset Discussion
The limitations in accuracies observed may be attributed to the lack of features in the dataset. In addition, there existed many trivial features that needed to be discarded (i.e. fire incident Id). Specifically for predicting the size of the fire, the quality of the features was lacking, as important factors, such as climate and humidity, were not included in the data set. Nonetheless, the benefit of this dataset was that it offered an opportunity to train on a large number of instances.


### Conclusion

Overall, the models provided a maximum accuracy of 52.9% for the size of the fire and 50.3% for the cause of the fire. Out of all of the models, the random forest approach provided the best accuracy of a little over 50% accross both datasets. It is important to note that using SVM and Neural Networks also resulted in similar accuracies around 50%, which establishes a trend accross these models. This 50% threshold indicates that there are factors that must be considered in terms of the input. For the UCI dataset, there was an abundance of features to train on. However, the limiting factor may have been the lack of instances to analyze, thereby not allowing for large-scale classifiers to be implemented. For the Kaggle dataset, the opposite is true where there existed a large quantity of instances, but there was a limited number of features, many of which were trivial and unrelated to the cause or size of the fire. In particular, the low accuracy of predicting the size of the fire can certainly be attributed to the quality of the features. Since the size is largely dependent on the climate and weather of the location, features that reflect these factors would improve the accuracy. Even with these limitations in mind, the overall achieved maximum accuracy of 52.9% is still significantly better than pure random guessing (total of 13 categories = 7.7%). The baseline qualifier of taking the most frequent cause of fires and using this as the prediciton resulted in an accuracy of 49%, which is still lower than the achieved accuracy. Similarly, for the fire size, the two baseline approaches gives accuracies of 14.3% and 48.3%. Given that an accuracies of 52.9% and 50.3% were achieved with the limited data to train on (either in terms of instances or features), the aforementioned improvements to the input data would potentially increase the accuracy even further.

### Bibliography
[1] “Facts + Statistics: Wildfires.” Insurance Information Institute, 2020, www.iii.org/fact-statistic/facts-statistics-wildfires. 

[2] Hausfather, Zeke. “Factcheck: How Global Warming Has Increased US Wildfires.” Carbon Brief, 20 Aug. 2018, www.carbonbrief.org/factcheck-how-global-warming-has-increased-us-wildfires. 

[3] Tatman, Rachael. “1.88 Million US Wildfires.” Kaggle, 2020, www.kaggle.com/rtatman/188-million-us-wildfires. 

[4] Castelli, Mauro, et al. “Predicting Burned Areas of Forest Fires: an Artificial Intelligence Approach.” Fire Ecology, vol. 11, no. 1, 2015, pp. 106–118., doi:10.4996/fireecology.1101106. 

[5] Chen, Jack, et al. “The FireWork v2.0 Air Quality Forecast System with Biomass Burning Emissions from the Canadian Forest Fire Emissions Prediction System v2.03.” Geoscientific Model Development, vol. 12, no. 7, 2019, pp. 3283–3310., doi:10.5194/gmd-12-3283-2019. 

[6] Potera, Carol. “CLIMATE CHANGE: Challenges of Predicting Wildfire Activity.” Environmental Health Perspectives, vol. 117, no. 7, July 2009, doi:10.1289/ehp.117-a293. 

[7] Subramanian, Sriram Ganapathi, and Mark Crowley. “Using Spatial Reinforcement Learning to Build Forest Wildfire Dynamics Models From Satellite Images.” Frontiers in ICT, vol. 5, 2018, doi:10.3389/fict.2018.00006. 

[8] Tonini, Marj, et al. “A Machine Learning-Based Approach for Wildfire Susceptibility Mapping. The Case Study of the Liguria Region in Italy.” Geosciences, vol. 10, no. 3, 2020, p. 105., doi:10.3390/geosciences10030105. 

[9] Rodrigues, Marcos, and Juan De La Riva. “An Insight into Machine-Learning Algorithms to Model Human-Caused Wildfire Occurrence.” Environmental Modelling &amp; Software, vol. 57, 2014, pp. 192–201., doi:10.1016/j.envsoft.2014.03.003. 

[10] Sayad, Younes Oulad, et al. “Predictive Modeling of Wildfires: A New Dataset and Machine Learning Approach.” Fire Safety Journal, vol. 104, 2019, pp. 130–146., doi:10.1016/j.firesaf.2019.01.006. 
