# Experimental Files for the ML4DQM/DC Project

This directory contains the files for implementing experiments in connection to the ML4DQM/DC Project. The goal of each experiment varies, but each is based at least partially or entirely on the Concatamash autoencoder. The different files and their functions are listed below

- `Permutations.py`/`Permutations.ipynb`: implementation of experiment to determine which histograms (if any) matter the most when trying to classify data as anomalous or not. The script takes a list of each different type of histogram available at the time of creation and goes through every combination of histograms (obeying certain rules such as 'Each subdetector must be invovled' or 'This histogram must be in every set') and creates a list of different histlists to iterate over. A group of concatamash autoencoders is trained on every histlist and the results are compared, sorted, and stored for evaluation. Different variations such as NLP and Fix exist only as .py files as well. These are essentially the same, but respectively use MSE working points instead of logarithmic probability or take in a list of labels for data to correct labels for the test set. 

- `TrainingPermutations.py`: implementation of experiment to determine if varying the training dataset changed the effectiveness of the autoencoders at classifying data. With recent information that many of the good/bad labels are unreliable, the answer to this question is very clear yes, so this script is more or less obsolete.

- `ProblemClass.ipynb`: implementation of experiment to see if autoencoders can determine what caused data to be labeled anomalous (such as which subdetector is at fault and what happened). This implementation iterates over different bad data in the test set and attempts to classify the purpose. 

All other files and directories are supports for the above experiments and either contain output data or extra input information.
