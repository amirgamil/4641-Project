# 4641-Project

### Introduction/Background

Wildfires are a major threat to our environment, destroying natural habitats and ravaging communities. In 2019, over 4.4 million acres were destroyed by wildfires, and 8 million acres have already been destroyed in 2020 alone [1]. Effects of climate change have resulted in greater wildfire activity over the past couple of years [2], and their ability to rapidly spread out of control puts great importance on extinguishing them as soon as possible. Rapidly detecting and predicting wildfires would allow first responders to act early and extinguish the fires, limiting their disastrous results.

### Problem definition

The goal of this project is to build a predictive model to understand, analyze, and detect patterns in data, allowing us to forecast the locations of wildfires as soon as possible. We have access to a dataset of 1.88 million wildfires in the United States [3], providing 24 years of geo-referenced wildfire records from 1992 to 2015. We are interested in exploring this dataset and applying machine learning techniques to identify patterns or clusters related to identifying causes of wildfires, predicting sizes of wildfires, or identifying wildfire "hotspots"(areas which are more prone to wildfires). The goal is to use machine learning techniques to aid in a real-world application of proper allocation of firefighter resources.

### Methods

Existing work attempts to use a variety of data, such as meteorological conditions [4], biomass [5], and satellite images [6]. However, there remain challenges in predicting wildfires accurately (i.e. predicting the complex patterns in how the environment will respond) [7].
  
Existing research in wildfire spread prediction incorporates high-dimensional features in classification. Tonini et al. use the slopes of land, vegetation type, non-flammable area, etc. to obtain the probability of fire in an Italian region via Random Forest with 15 years of fire damage maps were used as training data [8]. Final predictions ranged from 83.4% to 91.7% accuracy year-to-year under the test dataset. Rodrigues and Riva utilize features such as forest area, power line presence, protected area status, etc. to predict low/high risk in Spanish regions [9]. 30 years of wildfire data and used many regression methods - Random Forest, Boosted Regression Tree, Support Vector Machine, and logistic regression - to predict fire risk. The Random Forest algorithm proves promising with an AUC value of 0.746 vs 0.730, 0.709, and 0.686 for the other algorithms respectively. Sayad et al. use NASA satellite remote sensing data based on crop states, meteorological conditions, thermal intensity, etc. in conjunction with a Multi-layer perceptron neural network and a separate SVM model to predict fire occurrence to an accuracy of 98% and 97% respectively on a small dataset of Canadian fires [10].

We will utilize a correlation matrix between our features, followed by Principal Component Analysis to reduce our feature set to the most important, linearly independent features. To predict the wildfire risk of every location in our dataset, we will utilize an array of models - including Random Forest, SVM, and neural network-based classification. Each model will be tested independently to maximize accuracy.

### Potential results
We hope to apply a variety of machine learning techniques to determine features that best predict wildfires, enabling us to predict the probabilities of wildfires occurring within certain geographic clusters. Although it's difficult to say with certainty, we anticipate that variables utilized in previous research[4-6] will correlate highly with the presence of wildfires. This is a needle in the haystack problem in that we will need to filter out noise from the patterns we discover in our unsupervised/supervised learning methods to determine the features which are the best predictors of wildfires.

### Discussion

### Bibliography
[1] https://www.iii.org/fact-statistic/facts-statistics-wildfires#:~:text=The%20LNU%20Lightning%20Complex%20was,million%20acres%20burned%20in%202018.

[2] https://www.carbonbrief.org/factcheck-how-global-warming-has-increased-us-wildfires

[3] https://www.kaggle.com/rtatman/188-million-us-wildfires

[4] https://www.kaggle.com/rtatman/188-million-us-wildfires

[5] https://gmd.copernicus.org/articles/12/3283/2019/gmd-12-3283-2019.pdf

[6] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2717155/

[7] https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full

[8] https://www.mdpi.com/2076-3263/10/3/105

[9] https://www.sciencedirect.com/science/article/pii/S1364815214000814

[10] https://www.sciencedirect.com/science/article/pii/S0379711218303941
