# 4641-Project

Welcome to the best 4641 project in existence

Potential datasets:
https://www.kaggle.com/rtatman/188-million-us-wildfires
https://data.mendeley.com/datasets/85t28npyv7/1

### Introduction/Background

Wildfires cause irreversible damage to communities. The wildfires’ ability to spread at an exponential rate puts great importance on extinguishing them as soon as possible. As wildfires have become increasingly prevalent in recent years, we strive to help firefighters and communities by taking advantage of the unprecedented access to datasets and powerful technology.

### Problem definition

We have access to a dataset of 1.88 million wildfires in the United States, which provides 24 years of geo-referenced wildfire records, ranging from 1992 to 2015. We are interested in exploring this dataset and apply machine learning techniques to identify patterns or clusters which are related to identifying causes of a wildfire, the predicted size of a wildfire, or identifying "hotspots" for wildfires (areas which are more prone to wildfires than others). The goal is to use machine learning techniques to aid in a real-world application of proper allocation of firefighter resources.

### Methods

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

### Discussion
