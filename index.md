# 4641-Project

Welcome to the best 4641 project in existence

Dataset to start with:
https://www.kaggle.com/rtatman/188-million-us-wildfires

### Introduction/Background

Wildfires are a major threat to our environment, destroying natural habitats, threatening ecosystems, and ravaging communities. In 2019, over 4.4 million acres were destroyed by wildifires, and so far, 8 million acres have been destroyed in 2020 alone. [1]. Increased effects of climate change have resulted in greater wildfire activity over the past couple of years [2] and their ability to rapidly spread out of control puts great importance on extinguishing them as soon as possible. Rapidly detecting and predicting the locations of wildfires as early as possible would allow first responders to act early and extinguish wildfires, limiting their disastorous results.

### Problem definition

The goal of this project is to build a predictive model to understand, analyze, and detect patterns in data that can allow us to forecast the locations of wildfires as soon as possible. We have access to a dataset of 1.88 million wildfires in the United States [3] , which provides 24 years of geo-referenced wildfire records, ranging from 1992 to 2015. We are interested in exploring this dataset and apply machine learning techniques to identify patterns or clusters which are related to identifying causes of a wildfire, the predicted size of a wildfire, or identifying "hotspots" for wildfires (areas which are more prone to wildfires than others). The goal is to use machine learning techniques to aid in a real-world application of proper allocation of firefighter resources.

### Methods

Existing work in the past has attempted to use a variety of data, such as meterological conditions [4], biomass [5], and satellite images [6], however there remain pressing challenges in predicting wildfires accurately such as predicting the complex patterns in how the enivronment will respond [7].


----TODO:----
1. summarize existing literature (put together in actual paragraph)
2. explain specific methods we would use on potentially what - e.g. random forest with X data
  2a) first part can be unsupervised learning to understand patterns in data - say clustering (K-means, GMM, mention specifics) to find patterns between features, what's correlated with what
  2b) supervised learning using our results from ^ - try a variety of models e.g. random forests, SVM, neural networks to determine best fit
  
  
**A Machine Learning-Based Approach for Wildfire Susceptibility Mapping. The Case Study of the Liguria Region in Italy** 

- Uses many different features, including elevation, vegetation type, neighboring vegetation, and many more
- They modeled fires by binary classifying all areas as “burned” or not over a large amount of predictions. The burned vs total predictions ratio was used as a “probability” that a given area would be susceptible to wildfire.
- Primarily uses random forest (yeah, ikr) to make predictions
  -- Variables were optimized using Gini impurities
  -- Achieved ~80-90% accuracy

**An insight into machine-learning algorithms to model human-caused wildfire occurrence**

- Study in Spain using machine learning to predict fire risk (high vs low occurrence)
- Explantory variables – change in demographic potential, forest area, power line presence, railways, protected areas (hampers fire spreading)
- Tested with random forest, boosted regression tree, SVM, and logistic regression
  -- AUC values of .746, .730, .709, and .686 respectively

**Predictive modeling of wildfires: A new dataset and machine learning approach Younes Oulad Sayad, Hajar Mousannif, Hassan Al Moatassime**

- Utilizes remote sensing data (based on crop state, meteorological conditions)
  -- Utilizes data from NASA’s Terra, Aqua, Landsat, and Aster satellites, which can measure crop types, soil types, and thermal anomalies. We could use this
- Algorithms tested were Multi-layer perceptron neural network and SVM
- Pretty high accuracy (98% vs 97%), but vague solutions

**Algorithms**
The most successful models that I've encountered in the literature appear to be Random Forest for vanilla machine learning (upper 70's - 90 percent accuracy depending on the problem/solution types). Neural networks have the capacity to surpass this accuracy (90's percent range) but needs more research into this. Neural networks would also require many more features from what I gather.

### Potential results
We hope to apply a variety of machine learning techniques to determine features which are good predictors of wildfires and thus, can enable us to the probabilities of wildfires occuring within certain geographic clusters. Although it's difficult to say with certainty, we anticipate that the variables used in previous research in the field [4-6] will correlate highly with the presence of wildfires. This is a bit of a needle in the haystick problem in that we will need to filter out noise from the patterns we discover in our unsupervised/supervised learning methods to determine the features which are the best predictors of wildfires. 

### Discussion

### Bibliography
[1] https://www.iii.org/fact-statistic/facts-statistics-wildfires#:~:text=The%20LNU%20Lightning%20Complex%20was,million%20acres%20burned%20in%202018.
[2] https://www.carbonbrief.org/factcheck-how-global-warming-has-increased-us-wildfires
[3] https://www.kaggle.com/rtatman/188-million-us-wildfires
[4] https://www.kaggle.com/rtatman/188-million-us-wildfires
[5] https://gmd.copernicus.org/articles/12/3283/2019/gmd-12-3283-2019.pdf
[6] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2717155/
[7] https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full
