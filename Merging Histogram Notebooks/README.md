# Merging Histogram Techniques

This directory contains notebooks which demonstrate different methods of merging histograms. The different notebooks use unique approaches to combine different chunks of data together to get a more interesting dataset.
In more detail: 

- `1DMash.ipynb`: the original merging histograms concept. The 1D mash uses the 1D autoencoder developed by Jack Sisson as the basis for merging data. Essentially this notebook merges a given subset of the data together and trains an autoencoder for each subset. Ultimately this merging is controled by the histnames variable which contains a list of lists as explained in the tutorial. The sublist represents an indiviual autoencoder, and thus contains all the histograms going into a single model. The total list is the list of sublists. This notebook may provide more insight into how the merging concept works, but is ultimately rendered obsolete by Concatamash.

- `Concatamash.ipynb`: a sample notebook of how to create and train a Concatamash model. This model works on the same principle as the 1D mash, but instead combines the histograms in the training of the model. In this way, the preprocessing and postprocessing are nearly identical to the Combined autoencoder, but the training step is different. The concatenation is controlled in the same way as the 1DMash autoencoder, with the histlist. 

- `AEComparisons.ipynb`/`AEComparisons.py`: a python script and its notebook representation for comparing the different types of autoencoders. It uses common metrics to determine which autoencoder performed the best in testing and compares things such as training time. 