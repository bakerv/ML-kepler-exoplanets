# Machine Learning with Kepler Exoplanet Data
This project uses machine learning to predict the classification of celestial objects from the Kepler Exoplant Search Results dataset. Pandas, scikit-learn, and TensorFlow libraries for python are used.

[<img src="https://github.com/bakerv/ML-kepler-exoplanets/blob/main/images/kaggle_dataset.PNG">](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)


## Dataset

The [data](https://exoplanetarchive.ipac.caltech.edu/docs/data.html) used in this analysis comes from observations by NASA's [Kepler Space Telescope](https://en.wikipedia.org/wiki/Kepler_space_telescope), published by the NASA Exoplanet Science Institute.

The telescope was used to identify celstial objects that are potential candidates for earth sized planets orbiting other stars. These CANDIDATES were then later classified as CONFIRMED PLANETS or FALSE POSITIVES. 

 The goal of this analysis is to create a machine learning algorithm that will ingest the observation data, and correctly predict wether a candidate will be confirmed as an earth sized planet.


## Data preparation
The data needed to be carefully examined and prepared prior to model training. In particular rows that had not yet been given a definitive classification needed to be excluded from analysis.


[<img src="https://github.com/bakerv/ML-kepler-exoplanets/blob/main/images/labels.PNG">](https://github.com/bakerv/ML-kepler-exoplanets/blob/main/data_cleaning.ipynb)


Additionally, individual planet IDs and dates of observation were removed from the dataset. 

The numeric columns were scaled after splitting the data into training and validation sets. The categorical labels were given a binary classification of zero or one. 

 Shortly after beginning training on the initial model, the need to remove false positive flags to prevent data leakage became apparent. 

[<img src="https://github.com/bakerv/ML-kepler-exoplanets/blob/main/images/data_leakage.PNG">](https://github.com/bakerv/ML-kepler-exoplanets/blob/main/Model_1_false_flag_features_nn.ipynb)

 As seen in the above image, a model trained only on the false positive flags was able to achieve nearly perfect predictions after just one epoch. An indicator of poor data separation to be sure.

 ## Models
The primary models used for these analysis were deep neural networks, consisting of a single
hidden layer, built using the TensorFlow library.

 [<img src="https://github.com/bakerv/ML-kepler-exoplanets/blob/main/images/dnn_model.PNG">](https://github.com/bakerv/ML-kepler-exoplanets/blob/main/training_functions.py)

Hyperparameter tuning was carried out using [Hyperband](https://arxiv.org/abs/1603.06560) with the [Keras Tuner API](https://keras.io/keras_tuner/). This allowed for rapid unsupervised testing of hundreds of different model configurations. 

As a counterpoint to the complexity of training a neural network, random forest was used to quickly create a high quality predictive model.

[<img src="">](random_forest.ipynb)

 Random forest was implemented with the sklearn library using three lines of code, and a minimum of parameter configuration.

## Discussion

Overall, the deep neural network had better predictive accuracy. Random forest came in at 93%, while the dnn came in at 94.4%.

[<img src="">](model_2.ipynb)

One instance of hyperparameter searching came in with a model of over 95% accuracy. While this model was saved at checkpoint, it was unstable, an not able to be reproduced from the specified hyperparameter configuration. 

It is likely that more skilled tuning could result in better test accuracy from the neural network. In particular, learning rate schedulers and convolutional layers could be experimented with to reduce overfitting and improve feature detection. 

Even though it was not the absolute best, the random forest model was quite good. An added benefit of the model is the ease of feature identification. 

Once the columns containing false postive flags are removed from the dataset, we see that the margin of error for several values is a better predictor than the value itself. In particular, uncertainty in the transit duration, stellar effective temperature, and planetary radius are strong predictive measures; along with the planetary radius itself. 




## Sources
- “Kepler Space Telescope.” Wikipedia, Wikimedia Foundation, 19 Aug. 2021, en.wikipedia.org/wiki/Kepler_space_telescope. 
- Li, Lisha, et al. “Hyperband: A NOVEL BANDIT-BASED Approach to Hyperparameter Optimization.” ArXiv.org, 18 June 2018, arxiv.org/abs/1603.06560. 
- “NASA Exoplanet Archive.” NASA Exoplanet Archive, exoplanetarchive.ipac.caltech.edu/. 
- Nasa. “Kepler Exoplanet Search Results.” Kaggle, 10 Oct. 2017, www.kaggle.com/nasa/kepler-exoplanet-search-results. 
- Team, Keras. “Keras Documentation: Kerastuner.” Keras, keras.io/keras_tuner/. 