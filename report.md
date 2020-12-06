# Predicting Wildfire Clusters with Machine Learning

Group 11: Amir, Ryan, Sid, Yuma

### Video


### Introduction/Background

Wildfires are a major threat to our environment, destroying natural habitats and ravaging communities. In 2019, over 4.4 million acres were destroyed by wildfires, and 8 million acres have already been destroyed in 2020 alone [1]. Effects of climate change have resulted in greater wildfire activity over the past couple of years [2], and their ability to rapidly spread out of control puts great importance on extinguishing them as soon as possible. Rapidly detecting and predicting wildfires would allow first responders to act early and extinguish the fires, limiting their disastrous results.

### Problem definition

The goal of this project is to build a predictive model to understand, analyze, and detect patterns in data, allowing us to forecast the locations of wildfires as soon as possible. We have access to a dataset of 1.88 million wildfires in the United States [3], providing 24 years of geo-referenced wildfire records from 1992 to 2015. We are interested in exploring this dataset and applying machine learning techniques to identify patterns or clusters related to identifying causes of wildfires, predicting sizes of wildfires, or identifying wildfire "hotspots"(areas which are more prone to wildfires). The goal is to use machine learning techniques to aid in a real-world application of proper allocation of firefighter resources.

### Methods

Although previous attempts at wildfire predicted has useed a variety of data, such as meteorological conditions [4], biomass [5], and satellite images [6], there remain challenges in predicting wildfires accurately and understanding the complex patterns by which the environment will respond. [7].
  
Existing research in wildfire spread prediction incorporates high-dimensional features in classification. Tonini et al. use the slopes of land, vegetation type, non-flammable area, etc. to obtain the probability of fire in an Italian region via Random Forest with 15 years of fire damage maps were used as training data [8]. Final predictions ranged from 83.4% to 91.7% accuracy year-to-year under the test dataset. Rodrigues and Riva utilize features such as forest area, power line presence, protected area status, etc. to predict low/high risk in Spanish regions [9]. 30 years of wildfire data and used many regression methods - Random Forest, Boosted Regression Tree, Support Vector Machine, and logistic regression - to predict fire risk. The Random Forest algorithm proves promising with an AUC value of 0.746 vs 0.730, 0.709, and 0.686 for the other algorithms respectively. Sayad et al. use NASA satellite remote sensing data based on crop states, meteorological conditions, thermal intensity, etc. in conjunction with a Multi-layer perceptron neural network and a separate SVM model to predict fire occurrence to an accuracy of 98% and 97% respectively on a small dataset of Canadian fires [10].

We will utilize a correlation matrix between our features, followed by Principal Component Analysis to reduce our feature set to the most important, linearly independent features. To predict the wildfire risk of every location in our dataset, we will utilize an array of models - including Random Forest, SVM, and neural network-based classification. Each model will be tested independently to maximize accuracy.

### Results/Discussion
#### UCI Dataset
For each dataset, we attempted to classify our test wildfire data by fire size code (A - G). In our dataset containing Portuguese National Park fires, we observed a **52.9%** classification accuracy using a Random Forest classification model, **49.0%** accuracy with an AdaBoosted Random Forest classifier, **31.7%** accuracy with a Complement Naive Bayes classifier, **28.8%** accuracy with a Gaussian Naive Bayes classifier, **51.9%** accuracy with a Support Vector Machine classifier, and a **49.0%** accuracy with a Neural Network. With our dataset of United States wildfires, we approached 

A potential reason for our inconclusive classification results may lie in the size of the Portuguese National Park dataset. The dataset only contains 517 instances to analyze. It does, however, have a wide range of features to analyze. The bottleneck in our analysis may be lack of sufficent amounts of data to launch a large-scale classifier on. 

### Conclusion

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
