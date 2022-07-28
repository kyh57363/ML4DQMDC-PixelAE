### imports

# external modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
import importlib

# local modules
sys.path.append('../utils')
import csv_utils as csvu
import dataframe_utils as dfu
import hist_utils as hu
import autoencoder_utils as aeu
import plot_utils as pu
import generate_data_utils as gdu
importlib.reload(dfu)
importlib.reload(hu)
importlib.reload(aeu)
importlib.reload(pu)
importlib.reload(gdu)
sys.path.append('../src')
sys.path.append('../src/classifiers')
sys.path.append('../src/cloudfitters')
import DataLoader
importlib.reload(DataLoader)
import HistStruct
importlib.reload(HistStruct)
import AutoEncoder
importlib.reload(AutoEncoder)
import SeminormalFitter
import GaussianKdeFitter
import HyperRectangleFitter
importlib.reload(SeminormalFitter)
importlib.reload(GaussianKdeFitter)
importlib.reload(HyperRectangleFitter)
import HistStruct
importlib.reload(HistStruct)
import DataLoader
importlib.reload(DataLoader)
import AutoEncoder
importlib.reload(AutoEncoder)

class BuildAutoencoder:
    """A class to develop an object-oriented design for training and 
    developing autoencoders. The goal of this class is to improve the modularity
    of the current system in an easier to use way."""
    
    ### Defining attributes we'll need later
    
    ## A list storing a full version of the processed dataframe and one with only 
    # the specified runs
    dataframes = []
    
    ## A list of trained models for later reference
    models = []
    
    # A list of the training datasets for each model
    X_trainls = []
    
    # A list of lossplots from the training data for later reference about the model
    lossplots = []
    
    # A list which stores predictions for training data of each model
    predictionTrainls = []
    
    # A list of the MSEs of each model on the train data
    mseTrainls = []
    
    # Runs to be used in model training
    trainRuns = []
    