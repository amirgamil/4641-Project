# Midterm Report - Predicting Wildfire Clusters with Machine Learning

Group 11: Amir, Ryan, Sid, Yuma


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
![UCI Dataset](/report%20materials/uci_before_preprocessing.PNG "")
##### After Preprocessing
![UCI Dataset](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/uci_after_preprocessing.png)

#### [Kaggle Dataset](https://github.com/amirgamil/4641-Project/tree/master/datasets/kaggle)
The second dataset is a Kaggle dataset consisting of data from over 1.88 million wildfires in the United States. The dataset is initially stored in an SQLite database which we dump into Pandas in order to preprocess it easily. We perform 4 important steps on the dataset to preprocess it:

1. Convert latitude and longitude to standarized forms
2. Convert dates into timestamps with durations that can be processed by our models
3. Scale the columns which contain numerical data - we use Sci-kit learn's Standard Scaler to create Z-scores for each of our features
4. Convert string columns into categorical features
##### Before Preprocessing
![Kaggle Dataset](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_befre_preprocessing.png)
##### After Preprocessing
![Kaggle Dataset](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggledataset.png)

### Methods

Although previous attempts at wildfire predicted has useed a variety of data, such as meteorological conditions [4], biomass [5], and satellite images [6], there remain challenges in predicting wildfires accurately and understanding the complex patterns by which the environment will respond. [7].

Existing research in wildfire spread prediction incorporates high-dimensional features in classification. Tonini et al. use the slopes of land, vegetation type, non-flammable area, etc. to obtain the probability of fire in an Italian region via Random Forest with 15 years of fire damage maps were used as training data [8]. Final predictions ranged from 83.4% to 91.7% accuracy year-to-year under the test dataset. Rodrigues and Riva utilize features such as forest area, power line presence, protected area status, etc. to predict low/high risk in Spanish regions [9]. 30 years of wildfire data and used many regression methods - Random Forest, Boosted Regression Tree, Support Vector Machine, and logistic regression - to predict fire risk. The Random Forest algorithm proves promising with an AUC value of 0.746 vs 0.730, 0.709, and 0.686 for the other algorithms respectively. Sayad et al. use NASA satellite remote sensing data based on crop states, meteorological conditions, thermal intensity, etc. in conjunction with a Multi-layer perceptron neural network and a separate SVM model to predict fire occurrence to an accuracy of 98% and 97% respectively on a small dataset of Canadian fires [10].

Drawing on the above research, in order to build the most robust possible classification model, we divide our project into two steps: unsupervised learning in order to apply feature engineering and supervised learning where we test different classification models (e.g. Random Forests, Neural Networks, SVMs etc.) to find the most suited one for this task. In the feature engineering/unsupervised learning step, we:
1. Apply PCA to find the best, most relevant, linearly independent feautres
2. Build covariance/correlation matrices to understand the linear correlation between our different features

### Results/Discussion

### UCI Unsupervised Learning Results
#### Correlation Matrix
We started by building a correlation matrix, which depict the factors' correlations with each other via a gradient
![Correlation Matrix Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/uci_covariance.PNG)

Although most variables seem to be uncorrelated, there are some interesting findings. Firstly, feature X seems to be the most correlated feature with our groundtruth labels Y - with a positive correlation of around 0.54. This suggests that it will play an essential role when we do our downstream classification task and will aid the classification model greatly, especially since none of the other variables seem to be linearly correlated with our labels.  Secondly, features "FFMC", "DMC", "DC", "ISI", "temp" are highly correlated with each other. Intuitively, this makes sense since these features are all related with fuel and moisture content and thus, a change in one of them is likely to cause a change in the others. Thirdly, although there were no strong linear correlations between our features and label Y (besides the feature X), this does not necessarily mean that these features are not relevant or useful. One way we plan on exploring this is using neural networks which are good function approximators and may unconver higher order polynomial relationships between our features and our labels that can aid the classifcation model.

#### PCA Results
After building a correlation matrix, we perform Principal Component Analysis to reduce our feature set to the most important, linearly independent features. The figure below shows the results of our plots where we plot our data in the Z-space separated by its class label (different class labels correspond to different colors). ![PCA Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/uci_PCA.PNG)

Our results from PCA show that there is a lot of noise in our data. None of the classes were linearly inseparable, meaning that none of the features in the UCI dataset alone were strong predictors of our class labels. Naturally, this makes sense since wildfire intensity depends on many factors and the size of this dataset is relatively small. Because two dimensions were not enough to accurately represent our data, we will first plan on using all of our features then use backward selection with Lasso to select the most relevant features for our target classification task.


### Kaggle Unsupervised Learning Results
#### Correlation Matrix
Using the data from the Kaggle dataset, a correlation matrix was constructed to determine the relationship between the factors. 
![Correlation Matrix Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_covariance.PNG)

The correlation matrix depicts that there is, in fact, relationships between factors or a lack thereof. It is clear that there is one pair of factors that have a high correlation: "DISCOVERY_DOY" and "CONT_DOY". In fact, they have a perfect positive correlation of 1. These factors stand for the date of year in which the fire was discovered and contained, respectively. This is logical as the timing of the fire starting should be heavily related to when the fire is extinguished. Because of their high correlation, it would be wasteful to include both as this would require more space and time as they both represent the same data trends. Moreover, the "CONT_DOY" is dependent on the "DISCOVERY_DOY", and so keeping the "DISCOVERY_DOY" would be more optimal. Similarly, the correlation between "DISCOVERY_TIME" and "CONT_TIME" is 0.38. The factors stand for the time of day in which the fire was discovered and contained, repsectively. Again, these two factors would logically be related and thereby may not be necessary for including both in the analysis ("DISCOVERY_TIME" would be more favorable to keep following a similar reasoning to before). The correlation between "DISCOVERY_TIME" and "DISCOVERY_DOY" has the lowest absolute correlation of 0.04. This would indicate that it would be beneficial to keep both factors if they are correlated to the prediction of labels. Similarly, "DISCOVERY_TIME" and "FIRE_SIZE" have the second lowest absolute correlation of 0.014 thereby indicate a potential for the two factors to be beneficial to the prediction of labels. It is important here to note that regardless of a low absolute correlation, it is still crucial to determine the factors' relationship to the labels of interest as this correlation would help to further identify important factors. 

#### PCA Results
The data from the Kaggle dataset underwent a similar process to the UCI dataset. The result of the Principal Component Analysis is depicted in the plot below. The figure has 
two components that were constructed via PCA. (different class labels correspond to different colors). ![PCA Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_PCA.PNG)

The PCA of the Kaggle data set depicts two principal components and the casual factors as the labels. Altough there seems to be a trace of clustering, they are still not clear enough to the point where the model could be used to predict the casual factors. Indeed, the two components are only able to capture 35% of the variance. In order to capture 95% of the variance, eight components were necessary. In this way, it is clear that two components were not enough to represent the complexity of the dataset. Moreover, there may be some noise as aforementioned. Moving forward, we will utilize the eight components when implementing supervised learning. Although dimensionality reduction would help to mitigate risks of overfitting, beginning with the 95% of variance captured by the eight components would be more favorable as an initial step. Afterwards, if overfitting is indeed observed, the number of dimensions will be reduced as depicted by the PCA. During this iterative process, various approaches/models will be used to best classify the data. 

### Next Steps
Now that we have completed our unsupervised learning part of the project, we plan on moving on to the supervised portion where we will test different models to build a robust and accurate one at detecting wildfires. 

-----

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
