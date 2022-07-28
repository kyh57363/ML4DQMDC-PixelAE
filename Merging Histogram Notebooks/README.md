# Merging Histogram Techniques

This directory contains notebooks which demonstrate different methods of merging histograms. The different notebooks use unique approaches to combine different chunks of data together to get a more interesting dataset.
In more detail: 

- `1DMash.ipynb`: the original merging histograms concept. The 1D mash uses the 1D autoencoder developed by Jack Sisson as the basis for merging data. Essentially this notebook merges a given subset of the data together and trains an autoencoder for each subset. Ultimately this merging is controled by the histnames variable which contains a list of lists as explained in the tutorial. The sublist represents an indiviual autoencoder, and thus contains all the histograms going into a single model. The total list is the list of sublists. This notebook may provide more insight into how the merging concept works, but is ultimately rendered obsolete by Concatamash.

- `Concatamash.ipynb`: a sample notebook of how to create and train a Concatamash model. This model works on the same principle as the 1D mash, but instead combines the histograms in the training of the model. In this way, the preprocessing and postprocessing are nearly identical to the Combined autoencoder, but the training step is different. The concatenation is controlled in the same way as the 1DMash autoencoder, with the histlist. 

- `AEComparisons.ipynb`/`AEComparisons.py`: a python script and its notebook representation for comparing the different types of autoencoders. It uses common metrics to determine which autoencoder performed the best in testing and compares things such as training time. 


## Merging Concept

The concept of merging is essentially aimed at training fewer models. The goal is to save time and resource consumption without losing capability. There are currently two types of autoencoders which do not implement merging: 

### 1D Autoencoder

![alt text](https://github.com/kyh57363/ML4DQMDC-PixelAE/blob/master/Graphics/1D%20w%20Background.png?raw=true)

This is the method Jack Sisson created. As shown, the data is all immediately merged together and a single autoencoder is trained on the outcome. The training time is exceptional, but the loss of granularity and preprocessing required make this a poor choice. 

### Combined Autoencoder

![alt text](https://github.com/kyh57363/ML4DQMDC-PixelAE/blob/master/Graphics/Combined.png?raw=true)

This is the method implemented in the original reposity this one forks. The method trains an autoencoder on every histogram. There is virtually no preprocessing, but training time is long since there are many models.

### 1D Mash

![alt text](https://github.com/kyh57363/ML4DQMDC-PixelAE/blob/master/Graphics/1D%Mash.png.png?raw=true)

This was the first method at merging the two approaches, combining some histograms before training and then use the same method as combined to get the results. Though it restores some granularity from the 1D method and recovers training time over the Combined method, the 1D Mash has the greatest preprocessing overhead and is a little clunky. 

### Concatamash

![alt text](https://github.com/kyh57363/ML4DQMDC-PixelAE/blob/master/Graphics/Concatamash.png?raw=true)

This is the final and, so far, best implementation of merging histograms. The data flow is essentially the same as combined, but with the bonus of training fewer models. The preprocessing overhead is eliminated over the 1D Mash and the full granularity of Combined is possible with less training time. 
