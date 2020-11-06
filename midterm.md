# Midterm Report - Predicting Wildfire Clusters with Machine Learning

Group 11: Amir, Ryan, Sid, Yuma

### Introduction/Background

Wildfires are a major threat to our environment, destroying natural habitats and ravaging communities. In 2019, over 4.4 million acres were destroyed by wildfires, and 8 million acres have already been destroyed in 2020 alone [1]. Effects of climate change have resulted in greater wildfire activity over the past couple of years [2], and their ability to rapidly spread out of control puts great importance on extinguishing them as soon as possible. Rapidly detecting and predicting wildfires would allow first responders to act early and extinguish the fires, limiting their disastrous results.

### Problem definition

The goal of this project is to build a predictive model to understand, analyze, and detect patterns in data, allowing us to forecast the locations of wildfires as soon as possible. We have access to a dataset of 1.88 million wildfires in the United States [3], providing 24 years of geo-referenced wildfire records from 1992 to 2015. We are interested in exploring this dataset and applying machine learning techniques to identify patterns or clusters related to identifying causes of wildfires, predicting sizes of wildfires, or identifying wildfire "hotspots"(areas which are more prone to wildfires). The goal is to use machine learning techniques to aid in a real-world application of proper allocation of firefighter resources.

### Data Collection

We utilized a UCI dataset that consists of 517 entries with 13 features describing wildfire instances in a northeast Portuguese national park. These features include month, day, temperature, wind conditions, rain, etc. We use Pandas to preprocess our datasets and create two NxD matrices corresponding to the UCI and Kaggle dataset respectively. The purpose of this is to perform experiments first independently on each dataset to find the most relevant features for our classification target task.

#### Kaggle Dataset

We also utilized a kaggle dataset which consists of 1.88 million US wildfires spanning
over 24 years of geo-referenced wildfire records from 1992 to 2015 within the United
States. The kaggle dataset contains 39 features, but we decided that some of these
features do not add information as they pertain to unique identifiers which are meant
to be used in order to find more information on other reporting websites. While the
data found there might be useful, these identifiers are not useful for performing
unsupervised learning on this dataset specifically. Instead, we specifically selected
for columns which contain useful data regarding the fires, which reduces our feature set
from 39 features to 16 features.

##### Kaggle Data Cleaning

In this section we will summarize our efforts to clean the relevant features that
are contained in the kaggle dataset.

Our first step was to replace the FIRE_YEAR feature with a timestamp so we could
later perform regression using the timestamp as a time value. We were given the
year the fire took place along with the discovery day of year and specific discovery
time as 3 separate columns. We merged these 3 features into 1 feature for
DISCOVERY_TIMESTAMP, the timestamp at which the fire was discovered, and 1 feature for
CONTAINED_TIMESTAMP, the timestamp at which the fire was declared contained.

Note that we also chose to retain the Day of Year and Time features because those
could be used as continuous features from which a model could learn about fires
which happened at different times of years and different times of days. This would help
a model analyze patterns on fires which happened during the winter time versus summer
time and fires happening at night versus fires happening during the day.

Our next step was selecting the continuous features in order to properly standardize
those variables. Since PCA works well with standardized data, we wanted to
properly select the continuous features which could be properly used for dimensionality
reduction. We did not include categorical variables because those would influence the
dimensionality reduction algorithm when they should not be used for PCA.

### Methods

Remove?
Although previous attempts at wildfire predicted has useed a variety of data, such as meteorological conditions [4], biomass [5], and satellite images [6], there remain challenges in predicting wildfires accurately and understanding the complex patterns by which the environment will respond. [7].

TOREMOVE?
Existing research in wildfire spread prediction incorporates high-dimensional features in classification. Tonini et al. use the slopes of land, vegetation type, non-flammable area, etc. to obtain the probability of fire in an Italian region via Random Forest with 15 years of fire damage maps were used as training data [8]. Final predictions ranged from 83.4% to 91.7% accuracy year-to-year under the test dataset. Rodrigues and Riva utilize features such as forest area, power line presence, protected area status, etc. to predict low/high risk in Spanish regions [9]. 30 years of wildfire data and used many regression methods - Random Forest, Boosted Regression Tree, Support Vector Machine, and logistic regression - to predict fire risk. The Random Forest algorithm proves promising with an AUC value of 0.746 vs 0.730, 0.709, and 0.686 for the other algorithms respectively. Sayad et al. use NASA satellite remote sensing data based on crop states, meteorological conditions, thermal intensity, etc. in conjunction with a Multi-layer perceptron neural network and a separate SVM model to predict fire occurrence to an accuracy of 98% and 97% respectively on a small dataset of Canadian fires [10].

### Potential results/Discussion

### UCI Unsupervised Learning Results

#### Correlation Matrix

We started by building a correlation matrix. TODO: add analysis and pictures

#### PCA Results

After building a correlation matrix, we perform Principal Component Analysis to reduce our feature set to the most important, linearly independent features. The figure below shows the results of our plots where we plot our data in the Z-space separated by its class label (different class labels correspond to different colors). ![PCA Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/Screen%20Shot%202020-11-04%20at%208.41.24%20PM.png)

Our results from PCA show that there is a lot of noise in our data. None of the classes were linearly inseparable, meaning that none of the features in the UCI dataset alone were strong predictors of our class labels. Naturally, this makes sense since wildfire intensity depends on hundreds if not thousands of factors and this dataset represents only a very small subset of potential features. Due to the amount of noise in our data, our PCA results / correlation matrices have helped motivate the first line of attack when choosing models for our supervised learning, wildfire classification task. Specifically, we will start with Random Forests, training on around 80% of data and testing on the other 20%. It's very likely that given the amount of noise, our models will overfit. Thus, we will try to reduce overfitting by using the most relevant features we have obtained from our unsupervised learning, adding regularization, and trying a range of different models/ensemble learning to compare the classification accuracies.

### Kaggle Unsupervised Learning Results

#### Correlation Matrix

Using the data from the Kaggle dataset, a correlation matrix was constructed to determine the relationship between the factors. The correlation matrix depicts the factors' correlation via a gradient.
![Correlation Matrix Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_covariance.PNG)

![Correlation Matrix Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_covariance_more.PNG)

The correlation matrix depicts that there are, in fact, relationships between factors or a lack thereof. It is clear that there is one pair of factors that have a high correlation: "DISCOVERY_DOY" and "CONT_DOY". In fact, they have a perfect positive correlation of 1. These factors stand for the date of year in which the fire was discovered and contained, respectively. This is logical as the timing of the fire starting should be heavily related to when the fire is extinguished. Because of their high correlation, it would be wasteful to include both as this would require more space and time as they both represent the same data trends. Moreover, the "CONT_DOY" is dependent on the "DISCOVERY_DOY", and so keeping the "DISCOVERY_DOY" would be more optimal. Similarly, the correlation between "DISCOVERY_TIME" and "CONT_TIME" is 0.38. The factors stand for the time of day in which the fire was discovered and contained, repsectively. Again, these two factors would logically be related and thereby may not be necessary for including both in the analysis ("DISCOVERY_TIME" would be more favorable to keep following a similar reasoning to before). The correlation between "DISCOVERY_TIME" and "DISCOVERY_DOY" has the lowest absolute correlation of 0.04. This would indicate that it would be beneficial to keep both factors if they are correlated to the prediction of labels. Similarly, "DISCOVERY_TIME" and "FIRE_SIZE" have the second lowest absolute correlation of 0.014 thereby indicate a potential for the two factors to be beneficial to the prediction of labels. It is important here to note that regardless of a low absolute correlation, it is still crucial to determine the factors' relationship to the labels of interest as this correlation would help to further identify important factors.

#### PCA Results

The data from the Kaggle dataset underwent a similar process to the UCI dataset. The result of the Principal Component Analysis is depicted in the plot below. The figure has
two components that were constructed via PCA. (different class labels correspond to different colors). ![PCA Results](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_PCA.PNG)

The PCA of the Kaggle data set depicts two principal components and the casual factors as the labels. Altough there seems to be a trace of clustering, they are still not clear enough to the point where the model could be used to predict the casual factors. Indeed, the two components are only able to capture 36% of the variance. In order to capture 95% of the variance, eight components were necessary. This makes sense given our data cleaning process because the final dataset contained 9 features which were relatively distinct from each other. Thus, the PCA algorithm agreed with us and would have only
been able to condense/compress the information we supplied it into 8 components without
drastically compromising on information preserved.
In this way, it is clear that two components were not enough to represent the complexity of the dataset. Moreover, there may be some noise as aforementioned. Moving forward, we will utilize the eight components when implementing supervised learning. Although dimensionality reduction would help to mitigate risks of overfitting, beginning with the 95% of variance captured by the eight components would be more favorable as an initial step. Afterwards, if overfitting is indeed observed, the number of dimensions will be reduced as depicted by the PCA. During this iterative process, various approaches/models will be used to best classify the data.

We also performed a principal component analysis with 3 components and found that the retianed variance was higher but still only 53%.
![3D Kaggle PCA Results: Cause](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/3D_vis.png)

Additionally, we performed PCA but used States as the target variable,
which presented these graphs:

![Kaggle PCA Results: State](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_state_PCA.png)

![3D Kaggle PCA Results: State](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_state_PCA3D.png)

With larger samples taken:

![3D Kaggle PCA Results: State with larger sample](https://github.com/amirgamil/4641-Project/blob/master/report%20materials/kaggle_state_PCA3D_more.png)

In these cases, we note that some targets were clustered well using the PCAs
we performed on the kaggle datasets. This may indicate that while some fires
are unpredictable and have different natures, other fires might be predictable
for specific states. For example, California, a state known for having
many wildfires, was among those which had a predicatable clustering pattern.

//Remove?
We hope to apply a variety of machine learning techniques to determine features that best predict wildfires, enabling us to predict the probabilities of wildfires occurring within certain geographic clusters. Although it's difficult to say with certainty, we anticipate that variables utilized in previous research[4-6] will correlate highly with the presence of wildfires. This is a needle in the haystack problem in that we will need to filter out noise from the patterns we discover in our unsupervised/supervised learning methods to determine the features which are the best predictors of wildfires.

---

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
