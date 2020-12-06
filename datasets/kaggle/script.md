# What is Kaggle

The dataset is a Kaggle dataset consisting of data from over 1.88 million wildfires in the United States. The dataset is initially stored in an SQLite database which we dump into Pandas in order to preprocess it easily. We perform 4 important steps on the dataset to preprocess it:

1. Convert latitude and longitude to standarized forms
2. Convert dates into timestamps with durations that can be processed by our models
3. Scale the columns which contain numerical data - we use Sci-kit learn's Standard Scaler to create Z-scores for each of our features

# What is UCI

The UCI dataset consists of 517 entries with 13 features describing wildfire instances in a northeast Portuguese national park. These features include month, day, temperature, wind conditions, rain, etc. We perform several important steps on the data to preprocess it:

1. Convert year/month dates into categorical features
2. Scale the columns which contain numerical data (besides the dates which we leave as categorical data) - we use Sci-kit learn's Standard Scaler to create Z-scores for each of our features

# Our Methods

Although previous attempts at wildfire predicted has useed a variety of data, such as meteorological conditions [4], biomass [5], and satellite images [6], there remain challenges in predicting wildfires accurately and understanding the complex patterns by which the environment will respond. [7].

Existing research in wildfire spread prediction incorporates high-dimensional features in classification. In order to build the most robust possible classification model, we divide our project into two steps: unsupervised learning in order to apply feature engineering and supervised learning where we test different classification models to find the most suited one for this task. Our steps are:

1. Apply PCA to find the best, most relevant, linearly independent feautres
2. Build covariance/correlation matrices to understand the linear correlation between our different features
3. Develop a variety of machine learning models and neural networks to identify strengths and weaknesses of each.

## Unsupervised

### Correlation Matrices

For UCI, many variables seem to have no correlation. But, we note that feature X seems to be most correlated with our groundtruth labels Y with
a correlation of 0.54. Our other features "FFMC", "DMC", "DC", "ISI", "temp" are highly correlated with each other. However, we note that there
was no real correlation between these features and our groundtruth labels Y.

For Kaggle, there some relationships, but these we believe might not be useful for our classification and regression tasks. For example, we
see that there was little correlation between our features such as COUNTY, STATE, DISCOVERY, and CONTAINED to our FIRE_SIZE and FIRE_CAUSE truths.

With this information we hope to build models which can utilize whatever correlations there are in order to create robust models.

### PCA

Our UCI results from PCA show that there is a lot of noise in our data. None of the classes were linearly inseparable, meaning that none of the features in the UCI dataset alone were strong predictors of our class labels. Naturally, this makes sense since wildfire intensity depends on many factors and the size of this dataset is relatively small. Because two dimensions were not enough to accurately represent our data, we will first plan on using all of our features then use backward selection with Lasso to select the most relevant features for our target classification task.

The PCA of the Kaggle data set depicts two principal components and the casual factors as the labels. Altough there seems to be a trace of clustering, they are still not clear enough to the point where the model could be used to predict the casual factors. Indeed, the two components are only able to capture 36% of the variance. In order to capture 95% of the variance, eight components were necessary.

## Regression, Random forests, and Deep Learning

### Regression and Random Forests

We utilized a regression model to capture the possible linear relationships between our features of wildfires
and the eventual size of a fire. We believed that if there was such a relationship, then a regression model would
be sufficient to model that relationship to allow for robust predictions.

We utilized random forests as a collective of decision trees to enable good classification. We realized that singular
decision trees were prone to overfitting, so we opted for ensemble learning in order to create a robust classification model.

### Deep learning

We opted to use Deep learning for our project because we wanted to see if a Deep Learning model could
capture any complex relationships between the time and location of a a fire to its cause and eventual size.

While conventional machine learning models described above might be able to capture the relationships, we wanted to see
if a deep learning model could uncover more complicated patterns that the machine learning models couldn't.

---

Deep learning results

With our kaggle Neural network results, we found there we got around 47% accuracy on our test dataset of wildfires for classifying
the cause of a fire.

For kaggle regression, we got 5% accuracy to predict fire size.

### Random Forest

We utilized random forests due to the practicality provided by the limited number of features. This allowed for the drawback of typically high training times to be decreased significantly. Moreover, ensemble learning provided the means to mitigate potential overfitting. We implemented random forests to predict the cause and size of the fires.

---

Random Forest results

With the Kaggle data set, we achieved 44-48% accuracy on our test dataset for classifying the cause of the woldfires.

For predicting the size of the fire, the accuracy decreased to around 2.5%.

In both cases, the accuracy achieved reflects a similar trend to the results from deep learning. The accuracy for predicting the cause of the wildfire neared 50% while the accuracy for predicting the size of the fire was far less, about 5% at best.
