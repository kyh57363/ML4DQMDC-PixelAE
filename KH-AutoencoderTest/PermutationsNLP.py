
#!/usr/bin/env python
# coding: utf-8

# In[97]:


### imports

# external modules
import os
import gc
from os.path import exists
import os.path
import pandas as pd
from io import StringIO
import sys
from sys import getsizeof
import itertools
import numpy as np
import matplotlib.pyplot as plt
import importlib
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
import importlib
import psutil
import warnings
import nvidia_smi

# Necessary to keep GPU usage to a minimum
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# local modules
sys.path.append('../utils')
sys.path.append('./utils')
import csv_utils as csvu
import json_utils as jsonu
import dataframe_utils as dfu
import hist_utils as hu
import autoencoder_utils as aeu
import plot_utils as pu
import generate_data_utils as gdu
import refruns_utils as rru
importlib.reload(csvu)
importlib.reload(jsonu)
importlib.reload(dfu)
importlib.reload(hu)
importlib.reload(aeu)
importlib.reload(pu)
importlib.reload(gdu)
importlib.reload(rru)
sys.path.append('../src')
sys.path.append('./src')
sys.path.append('../src/classifiers')
sys.path.append('./src/classifiers')
sys.path.append('../src/cloudfitters')
sys.path.append('./src/cloudfitters')
import HistStruct
importlib.reload(HistStruct)
import FlexiStruct
importlib.reload(FlexiStruct)
import DataLoader
importlib.reload(DataLoader)
import AutoEncoder
importlib.reload(AutoEncoder)
import SeminormalFitter
import GaussianKdeFitter
import HyperRectangleFitter
importlib.reload(SeminormalFitter)
importlib.reload(GaussianKdeFitter)
importlib.reload(HyperRectangleFitter)


# In[74]:

# Data selection
year = '2017'
eras = ['E', 'F']

datadir = {}
for era in eras:
    datadir[year + era] = '../data/' + year + era + '/'


# In[75]:

# Histogram type selection
blk1Vars = ['chargeInner', 'chargeOuter', 'adc', 'size']
blk2Vars = ['NormalizedHitResiduals', 
            'Summary_ClusterStoNCorr__OnTrack_',
            #'Summary_TotalNumberOfDigis_'
           ]
blk3Vars = [
    #'NumberOfTracks', 
    'NumberOfRecHitsPerTrack', 
    'Chi2oNDF',
    'goodvtxNbr'] 
miscVars = [
    #'NumberOfClustersInPixel', 
    'num_clusters_ontrack_PXBarrel', 
    'num_clusters_ontrack_PXForward', 
    #'NumberOfClustersInStrip'
]


# In[76]:


### Get the different permutations for block 1
combosBlk1 = []
histcount = 0
modelcount = 0
for size in range(1, len(blk1Vars) + 1):
    # Get every combination of given size
    thisList = list(itertools.combinations(blk1Vars, size))
    
    
    ## Applying rules
    for item in thisList:
        
        if 'chargeInner' not in item: continue
        
        subList = []
        subListPX = []
        subListDSP = []
        subListDSN = []
        # Getting individual histograms to set appropriate names
        for element in item:
            
            # Need to treat chargeInner the same as 'charge' for disks
            if element == 'chargeInner':
                for i in range(1, 4):
                    subListDSP.append('charge_PXDisk_+' + str(i))
                    subListDSN.append('charge_PXDisk_-' + str(i))
                    
            elif element != 'chargeOuter':
                for i in range(1, 4):
                    subListDSP.append(element + '_PXDisk_+' + str(i))
                    subListDSN.append(element + '_PXDisk_-' + str(i))
            # PXlayers
            for i in range(1, 5):
                subListPX.append(element + '_PXLayer_' + str(i))
        
        subList.append(subListPX)
        subList.append(subListDSP)
        subList.append(subListDSN)
        
        combosBlk1.append(subList)


# In[77]:


### Permutations for block 2
combosBlk2 = []
for size in range(1, len(blk2Vars) + 1):
    # Get every combination of given size
    thisList = list(itertools.combinations(blk2Vars, size))
    
    
    ## Applying rules
    for item in thisList:
        if 'Summary_ClusterStoNCorr__OnTrack_' not in item: continue
            
        subList = []
        subTIB = []
        subTOB = []
        subTIDP = []
        subTIDN = []
        subTECP = []
        subTECN = []
        
        # Getting individual histograms to set appropriate names
        for element in item:
            
            # Special case
            if element != 'NormalizedHitResiduals':
                for i in range(1, 10):
                    subTECN.append(element + '_TEC__MINUS__wheel__' + str(i))
                    subTECP.append(element + '_TEC__PLUS__wheel__' + str(i))
                    
                for i in range(1, 4):
                    subTIDN.append(element + '_TID__MINUS__wheel__' + str(i))
                    subTIDP.append(element + '_TID__PLUS__wheel__' + str(i))
                    
                for i in range(1, 5):
                    subTIB.append(element + '_TIB__layer__' + str(i))
            
                for i in range(1, 7):
                    subTOB.append(element + '_TOB__layer__' + str(i))
                    
            else: 
                for i  in range(1, 10):
                    subTECN.append(element + '_TEC__wheel__' + str(i))
                    
                for i in range(1, 4):
                    subTIDN.append(element + '_TID__wheel__' + str(i))
                    
                for i in range(1, 5):
                    subTIB.append(element + '_TIB__Layer__' + str(i))
            
                for i in range(1, 7):
                    subTOB.append(element + '_TOB__Layer__' + str(i))

        subList.append(subTIB)
        subList.append(subTOB)
        if len(subTIDP) > 0:
            subList.append(subTIDP)
        subList.append(subTIDN)
        if len(subTECP) > 0:
            subList.append(subTECP)
        subList.append(subTECN)
        combosBlk2.append(subList)


# In[78]:


### Permutations for block 3
combosBlk3 = []

for size in range(1, len(blk3Vars) + 1):
    # Get every combination of given size
    thisList = list(itertools.combinations(blk3Vars, size))
    
    ## Applying rules
    for item in thisList:
        if 'NumberOfRecHitsPerTrack' not in item: continue
         
        subList = []
        for element in item:
            if element !='goodvtxNbr':
                subList.append(element + '_lumiFlag_GenTk')
            else:
                subList.append(element)
        
        combosBlk3.append([subList])


# In[79]:


### Permutations for block 4
combosBlk4 = []

for size in range(0, len(miscVars) + 1):
    # Get every combination of given size
    thisList = list(itertools.combinations(miscVars, size))
    
    ## Applying rules
    for item in thisList:
        subList = []
        subSubList = []
        for element in item:
            subSubList.append(element)
    
        subList.append(subSubList)
        combosBlk4.append(subList)


# In[80]:


### Parsing combinations to create histlists
histlists = []
conmodelcount = 0
combmodelcount = 0
for combo1 in combosBlk1:
    for combo2 in combosBlk2:
        for combo3 in combosBlk3:
            for combo4 in combosBlk4:
                curList = []
                for element in combo1:
                    curList.append(element)
                for element in combo2:
                    curList.append(element)
                for element in combo3:
                    curList.append(element)
                for element in combo4:
                    if len(element) > 0:
                        curList.append(element)
                
                # Sanity check that all files exist
                for histgroup in curList:
                    for hist in histgroup:
                        for era in eras:
                            filename = 'DF' + year + era + '_' + hist +'.csv'
                            path = datadir[year + era] + filename
                            if not os.path.exists(path):
                                raise Exception('Histogram {} does not exist!'.format(hist))
                histlists.append(curList)
                
for histlist in histlists:
    for histgroup in histlist:
        conmodelcount = conmodelcount + 1
        for hist in histgroup:
            combmodelcount = combmodelcount + 1
            
print('Models to Train:')
print(' - Concatamash: ' + str(conmodelcount))

print('\nTraining Sets: ' + str(len(histlists)))
print()


# In[81]:


### Data Controls and Selection - 1D Autoncoder

# select a list of good runs to test on in development training_mode
# should be validated by eye
trainrunsls = {"2017E":
     {
        #   "303819":[[-1]],
        #   "303999":[[-1]],
        #   "304119":[[-1]],
        #   "304120":[[-1]],
        #   "304197":[[-1]],
          "304505":[[-1]],
          "304198":[[-1]],
          "304199":[[-1]],
          "304209":[[-1]],
          "304333":[[-1]],
          "304446":[[-1]],
          "304449":[[-1]],
          "304452":[[-1]],
          "304508":[[-1]],
          "304625":[[-1]],
          "304655":[[-1]],
          "304737":[[-1]],
          "304778":[[-1]],
          "306459":[[-1]],
          "304196":[[-1]],
     },"2017F":
     {
        #   "305310":[[-1]],
        #   "305040":[[-1]],
        #   "305043":[[-1]],
        #   "305185":[[-1]],
        #   "305204":[[-1]],
          "305234":[[-1]],
          "305247":[[-1]],
          "305313":[[-1]],
          "305338":[[-1]],
          "305350":[[-1]],
          "305364":[[-1]],
          "305376":[[-1]],
          "306042":[[-1]],
          "306051":[[-1]],
          "305406":[[-1]],
          "306122":[[-1]],
          "306134":[[-1]],
          "306137":[[-1]],
          "306154":[[-1]],
          "306170":[[-1]],
          "306417":[[-1]],
          "306432":[[-1]],
          "306456":[[-1]],
          "305516":[[-1]],
          "305586":[[-1]],
          "305588":[[-1]],
          "305590":[[-1]],
          "305809":[[-1]],
          "305832":[[-1]],
          "305840":[[-1]],
          "305898":[[-1]],
          "306029":[[-1]],
          "306037":[[-1]],
          "306095":[[-1]],
     },
}

goodrunsls = {'2017E': 
{
          "303819":[[-1]],
          "303999":[[-1]],
          "304119":[[-1]],
          "304120":[[-1]],
          "304197":[[-1]],
        #   "304505":[[-1]],
        #   "304198":[[-1]],
        #   "304199":[[-1]],
        #   "304209":[[-1]],
        #   "304333":[[-1]],
        #   "304446":[[-1]],
        #   "304449":[[-1]],
        #   "304452":[[-1]],
        #   "304508":[[-1]],
        #   "304625":[[-1]],
        #   "304655":[[-1]],
        #   "304737":[[-1]],
        #   "304778":[[-1]],
        #   "306459":[[-1]],
        #   "304196":[[-1]],
     },"2017F":
     {
          "305310":[[-1]],
          "305040":[[-1]],
          "305043":[[-1]],
          "305185":[[-1]],
          "305204":[[-1]],
        #   "305234":[[-1]],
        #   "305247":[[-1]],
        #   "305313":[[-1]],
        #   "305338":[[-1]],
        #   "305350":[[-1]],
        #   "305364":[[-1]],
        #   "305376":[[-1]],
        #   "306042":[[-1]],
        #   "306051":[[-1]],
        #   "305406":[[-1]],
        #   "306122":[[-1]],
        #   "306134":[[-1]],
        #   "306137":[[-1]],
        #   "306154":[[-1]],
        #   "306170":[[-1]],
        #   "306417":[[-1]],
        #   "306432":[[-1]],
        #   "306456":[[-1]],
        #   "305516":[[-1]],
        #   "305586":[[-1]],
        #   "305588":[[-1]],
        #   "305590":[[-1]],
        #   "305809":[[-1]],
        #   "305832":[[-1]],
        #   "305840":[[-1]],
        #   "305898":[[-1]],
        #   "306029":[[-1]],
        #   "306037":[[-1]],
        #   "306095":[[-1]],
     }, 

}

badrunsls = {"2017E":
     {
          "303824":[[-1]],
          "303989":[[-1]],
          "304740":[[-1]],
     },"2017F":
     {
          "305250":[[-1]],
          "306422":[[-1]],
          "305249":[[-1]],
     },
}

# Gets the full list of histograms to be added to the HistStruct
histnames = histlists[255] 


# In[82]:


### Define Run Properties - Combined Autoencoder
# in this cell all major run properties are going to be set,

# Select whether to save a new histstruct
save = False


# In[83]:


### Model Controls and Selection - 1D Autoencoder

plotNames = 'Test'
name = plotNames+'plots'

# Choose whether to train a new model or load one
trainnew = True
savemodel = True
modelname = plotNames

# Bias Factors
fmBiasFactor = 2
wpBiasFactor = 2


# In[84]:


# train on a user-defined subset of runs
    
# Select runs to be used in training from the user-defined list
runsls_training = {}
runsls_bad = {}
runsls_good = {}
for era in eras:
    runsls_training.update(trainrunsls[year + era])
    # Select bad runs to test on in the user-defined list
    runsls_bad.update(badrunsls[year + era])
    # Select good runs to test on in the user-defined list
    runsls_good.update(goodrunsls[year + era])


# In[85]:


# Initializations
dloader = DataLoader.DataLoader()
histstruct = FlexiStruct.FlexiStruct()
histstruct.reset_histlist(histnames)
failedruns = {}
failedls ={}
# Unpack histnames and add every histogram individually
consistent = True
#sys.sdout = open('HistPerm.log' , 'w')
for era in eras:
    for histnamegroup in histnames:
        for histname in histnamegroup:
            sys.stdout.write('\rAdding {}...'.format(histname + era) + '                                                ')
            sys.stdout.flush()       
    
            # Bring the histograms into memory from storage for later use
            filename = datadir[year + era] + 'DF' + year + era + '_' + histname + '.csv'
            df = dloader.get_dataframe_from_file( filename )
            
            # In case of local training, we can remove most of the histograms
            if( runsls_training is not None and runsls_good is not None and runsls_bad is not None ):
                runsls_total = {k: v for d in (runsls_training, runsls_good, runsls_bad) for k, v in d.items()}
                df = dfu.select_runsls( df, runsls_total )
            
            df = dfu.rm_duplicates(df)
            
            try:
                # Store the data in the histstruct object managing this whole thing
                orig_out = sys.stdout
                sys.stdout = StringIO()
                histstruct.add_dataframe( df, rebinningfactor = 1, standardbincount = 102 )
                sys.stdout = orig_out
            except:
                sys.stdout = orig_out
                print("WARNING: Could not add " + histname, file=sys.stderr)
                failedruns[histname] = dfu.get_runs(df)
                failedls[histname] = dfu.get_ls(df)
                consistent = False
            
sys.stdout.write('\rData import complete.')
sys.stdout.flush()
#sys.sdout.close()

# In[86]:


runsls_total = {k: v for d in (runsls_training, runsls_good, runsls_bad) for k, v in d.items()}
inconsistentRuns = {}
if not consistent:
    for histname in failedruns:
        inconsistentRuns[histname] = {}
        for run in runsls_total:
            if int(run) not in failedruns[histname]:
                runls = {}
                runls[run] = [[-1]]
                inconsistentRuns[histname].update(runls)
    print(inconsistentRuns)


# In[87]:


### Add Masks to Data
def assignMasks(histstruct, runsls_training, runsls_good, runsls_bad):

    histstruct.add_dcsonjson_mask( 'dcson' )
    histstruct.add_goldenjson_mask('golden' )
    histstruct.add_highstat_mask( 'highstat', entries_to_bins_ratio=0)
    histstruct.add_stat_mask( 'lowstat', max_entries_to_bins_ratio=0)
    if runsls_training is not None: histstruct.add_json_mask( 'training', runsls_training )
    if runsls_good is not None: histstruct.add_json_mask( 'good', runsls_good )
        
    # Distinguishing bad runs
    nbadruns = 0
    if runsls_bad is not None:
        histstruct.add_json_mask( 'bad', runsls_bad )
        
        # Special case for bad runs: add a mask per run (different bad runs have different characteristics)
        nbadruns = len(runsls_bad.keys())
        for i,badrun in enumerate(runsls_bad.keys()):
            histstruct.add_json_mask( 'bad{}'.format(i), {badrun:runsls_bad[badrun]} )
            
    if save:
        histstruct.save('test.pk1')
    return histstruct


# In[88]:


def define_concatamash_autoencoder(histstruct):
    
    histslist = []
    vallist = []
    autoencoders = []
    if trainnew:
        for i,histnamegroup in enumerate(histnames):
            
            train_normhist = np.array([hu.normalize(histstruct.get_histograms(
                histname = hname, masknames = ['dcson','highstat', 'training']), 
                                                 norm="l1", axis=1) 
                                       for hname in histnamegroup]).transpose((1,0,2))
            X_train, X_val = train_test_split(train_normhist, test_size=0.4, random_state=42)
            
            # Half the total bin count
            arch = 51 * len(histnamegroup)
            
            ## Model parameters
            input_dim = X_train.shape[2] #num of predictor variables
            Input_layers=[Input(shape=input_dim) for i in range((X_train.shape[1]))]
            
            # Defining layers
            conc_layer = Concatenate()(Input_layers)
            encoder = Dense(arch * 2, activation="tanh")(conc_layer)
            encoder = Dense(arch/2, activation='relu')(encoder)
            encoder = Dense(arch/8, activation='relu')(encoder)
            encoder = Dense(arch/16, activation='relu')(encoder)
            decoder = Dense(arch/8, activation="relu")(encoder)
            decoder = Dense(arch/2, activation='relu')(encoder)
            decoder = Dense(arch * 2, activation="tanh")(decoder)
            
            Output_layers=[Dense(input_dim, activation="tanh")(decoder) for i in range(X_train.shape[1])]

            autoencoder = Model(inputs=Input_layers, outputs=Output_layers)
            autoencoders.append(autoencoder)
            
            histslist.append(X_train)
            vallist.append(X_val)
     
    # Return the histograms stored 2-Dimensionally and the autoencoders corresponding
    return(histslist, vallist, autoencoders, train_normhist)


# In[89]:


### Trains a combined autoencoder for every merge set
def train_concatamash_autoencoder(histstruct, histslist, vallist, autoencoders):
    
    # Iterate through the training data to train corresponding autoencoders
    autoencodersTrain = []
    for i in range(len(histslist)):
        
        sys.stdout.write('\rNow training model {}/'.format(i + 1) + str(len(histslist)))
        sys.stdout.flush()
        
        # Set variables to temporary values for better transparency
        X_train = histslist[i]
        X_val = vallist[i]
        autoencoder = autoencoders[i]
        
        
        ## Model parameters
        nb_epoch = 1000
        batch_size = 10000
        
        #checkpoint_filepath = 'checkpoint'
        #model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #   filepath=checkpoint_filepath,
        #   save_weights_only=False,
        #   verbose=1,
        #   save_best_only=True,
        #   monitor='val_loss',
        #   mode='min')
        
        # Tell the model when to stop
        earlystop = EarlyStopping(monitor='val_loss',
            min_delta=1e-7,
            patience=50,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
        )
        lr =0.001
        opt = keras.optimizers.Adam(learning_rate=lr)
        
        autoencoder.compile(loss='mse',
                            optimizer=opt)
        
        ## Train autoencoder
        train = autoencoder.fit(x=[X_train[:,i] for i in range(X_train.shape[1])],
                                y=[X_train[:,i] for i in range(X_train.shape[1])],
                                epochs=nb_epoch,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_data=([X_val[:,i] for i in range(X_val.shape[1])], [X_val[:,i] for i in range(X_val.shape[1])]),
                                verbose=0,
                                callbacks= [earlystop],    
                                )
        
        # Save classifier for evaluation
        classifier = AutoEncoder.AutoEncoder(model=autoencoder)
        histstruct.add_classifier(histnames[i][0], classifier) 
        autoencodersTrain.append(classifier)
        K.clear_session()
        del(autoencoder, classifier)
    return autoencodersTrain


# In[90]:


### Evaluate the Models for WP definition
def evaluate_models_train(histstruct):
    
    for histgroup in histnames:
        histstruct.evaluate_classifier(histgroup)
    
    # get mse for training set
    if 'training' in histstruct.masks.keys(): masknames = ['dcson','highstat', 'training']
    else: masknames = ['dcson','highstat']
    mse_train = histstruct.get_scores_array( masknames=masknames )
    
    # get mse for good set
    if 'good' in histstruct.masks.keys():
        mse_good = []
        for histgroup in histnames:
            for histname in histgroup:
                mse_good.append(histstruct.get_scores( histname=histname, masknames=['dcson','highstat','good'] ))
    else:
        mse_good = []
        for histgroup in histnames:
            for histname in histgroup:
                hists_good = hu.averagehists( histstruct.get_histograms( histname=histname, masknames=['dcson','highstat']), 1000 )
                thismse = histstruct.classifiers[histname].evaluate( hists_good )
                mse_good.append( thismse )
                print(run)
    mse_good = np.array(mse_good)
    mse_good = np.transpose(mse_good)
    
    # get mse for bad sets
    mse_bad = []
    nbadruns = len([name for name in runsls_bad])
    for i in range(nbadruns):
        mse_bad.append( histstruct.get_scores_array( masknames=['dcson','highstat','bad{}'.format(i)] ) )
        
    return [mse_train, mse_good, mse_bad]


# In[91]:


### Plots and Distribution Analysis
def fit_mse_distribution(mse_good, mse_bad_list, wpBiasFactor):
    
    # Reshaping arrays to be easily averageable
    mseArrays = []
    for i in range(mse_good.shape[1]):
        mseArrays.append(np.zeros(mse_good.shape[0]))
    
    for i,mseSet in enumerate(mse_good):
        for j,mseArray in enumerate(mseArrays):
            mseArray[i] = mseSet[j]
    
    # Pulling out the average good values
    avgGood = []
    for mseArray in mseArrays:
        avgGood.append(np.mean(mseArray))
    avgGood = np.array(avgGood)
    
    # Reshaping bad arrays to be easily averageable - this is where future implementations for picking out specific problems in data would be
    avgBadList = []
    for mse_bad in mse_bad_list:
        mseArrays = []
        for i in range(mse_bad.shape[1]):
            mseArrays.append(np.zeros(mse_bad.shape[0]))
        for i,mseSet in enumerate(mse_bad):
            for j,mseArray in enumerate(mseArrays):
                mseArray[i] = mseSet[j]
        avgBad = []
        for mseArray in mseArrays:
            avgBad.append(np.mean(mseArray))
        
        avgBadList.append(avgBad)
    
    # Getting averages of bads
    avgBads = np.zeros((mse_good.shape[1], len(avgBadList)))
    print(avgBads.shape)
    for i in range(len(avgBadList)):
        for j in range(mse_good.shape[1]):
            avgBads[j][i] = avgBadList[i][j]
    avgBadsTot = np.zeros(mse_good.shape[1])
    for i in range(mse_good.shape[1]):
        avgBadsTot[i] = np.mean(avgBads[i])
        
    # Finding working points for each individual histogram (the simple approach)
    workingPoints = np.zeros(mse_good.shape[1])
    for i in range(len(avgBadsTot)):
        workingPoints[i] = (1 / (1 + wpBiasFactor)) * (wpBiasFactor * avgGood[i] + avgBadsTot[i])

    sep = np.mean(avgBadsTot) - np.mean(avgGood)     
    
    return [workingPoints, avgGood, avgBadsTot], sep


# In[93]:


### Prepare MSEs for Working Point Definition
# - Note: This test is geared towards testing. Updates are required
#       for use with unlabeled data
def evaluate_autoencoders_combined(mse_good_eval, mse_bad_list, wpData, wp_test_split):
    workingPoints = wpData[0]
    
    # Prepare data for comparison to WP
    mse_good_test = np.array(mse_good_eval[int(wp_test_split * len(mse_good_eval)):])
    mse_bad_eval = np.concatenate(tuple(mse_bad_list))
    mse_bad_test = mse_bad_eval[int(wp_test_split * len(mse_bad_eval)):]
    
    labels_good = np.zeros(len(mse_good_test))
    labels_bad = np.ones(len(mse_bad_test))
    
    # Percentage of histograms with issues to flag an anomaly
    testWeight = 10
    # Check each histogram and if a sufficient number are anomalous, flag the lumisection
    anomalousList = []
    preds_good = np.zeros(len(mse_good_test))
    for i,lumi in enumerate(mse_good_test):
        values = np.ones(len(lumi[lumi > workingPoints]))
        if np.sum(values) > (testWeight/100) * len(lumi):
            preds_good[i] = 1
    
    # Same, but with known bad data
    preds_bad = np.zeros(len(mse_bad_test))
    for i, lumi in enumerate(mse_bad_test):
        values = np.ones(len(lumi[lumi < workingPoints]))
        anomalousList.append(np.where(lumi > workingPoints, True, False))
        if np.sum(values) > (testWeight/100) * len(lumi):
            preds_bad[i] = 1
            
    labels = np.concatenate(tuple([labels_good, labels_bad]))
    scores = np.concatenate(tuple([preds_good, preds_bad]))
    
    (tp, fp, tn, fn) = get_confusion_matrix(scores,labels)

    # Get metrics for analysis
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if np.isnan(precision): precision = 0
    if np.isnan(recall): recall = 0
    try:
        f_measure = (1 + fmBiasFactor * fmBiasFactor) * ((precision * recall) / ((fmBiasFactor * fmBiasFactor * precision) + recall)) 
    except RuntimeWarning:
        f_measure = 0
    if np.isnan(f_measure): f_measure = 0

    return (f_measure, accuracy, precision, recall)

# In[94]:


def get_confusion_matrix(scores, labels):
    ### plot a confusion matrix
    # input arguments:
    # - scores and labels: defined in the same way as for get_roc
    # - wp: the chosen working point 
    #       (i.e. any score above wp is flagged as signal, any below is flagged as background)
    #       note: wp can be a integer or float, in which case that value will be used directly,
    #             or it can be a string in which case it will be used as the 'method' argument in get_wp!
    # - plotwp: only relevant if wp is a string (see above), in which case plotwp will be used as the 'doplot' argument in get_wp

    nsig = np.sum(labels)
    nback = np.sum(1-labels)

    # get confusion matrix entries
    tp = np.sum(np.where((labels==1) & (scores==1),1,0))/nsig
    fp = np.sum(np.where((labels==0) & (scores==0),1,0))/nback
    tn = 1-fp
    fn = 1-tp

    
    # old plotting method with seaborn
    #df_cm = pd.DataFrame(cmat, index = [true_negative_label,true_positive_label],
    #              columns = [predicted_negative_label,predicted_positive_label])
    #fig,ax = plt.subplots()
    #sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    
    # new plotting method with pyplot

    # printouts for testing
    #print('working point: {}'.format(wp))
    #print('nsig: {}'.format(nsig))
    #print('nback: {}'.format(nback))
    #print('true positive / nsig: {}'.format(tp))
    #print('false positive / nback: {}'.format(fp))

    # return the working point (for later use if it was automatically calculated)
    return (tp, fp, tn, fn)


# In[95]:


### Function to print memory information for debugging memory leaks
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# In[100]:


### Loop it Fxn
def masterLoop(aeStats, numModels, histnames, histstruct, debug):
    try:
        percComp = (numModels/conmodelcount)*100
        print('Running Job {}/'.format(i+1) + str(len(histlists)) + ' - {:.2f}% Complete'.format(percComp))
        
        # Update histlist to reflect new data
        histstruct.reset_histlist(histnames, suppress=True)
        assignMasks(histstruct, runsls_training, runsls_good, runsls_bad)
        
        # Build autoencoders based on new data
        (histslist, vallist, autoencoders, train_normhist) = define_concatamash_autoencoder(histstruct)
        
        # Train autoencoders based on current histlist and record speed
        start = time.perf_counter()
        
        # Suppres Unhelpful Error Messages
        orig_out = sys.stderr
        sys.stderr = open('trash', 'w')
        autoencoders = train_concatamash_autoencoder(histstruct, histslist, vallist, autoencoders)
        sys.stderr = orig_out
        
        numModels += len(autoencoders)
        
        stop = time.perf_counter()
        trainTime = stop - start
        
        sys.stdout.write('\rTraining complete in ' + str(trainTime) + ' seconds')
        sys.stdout.flush()
        print()
        
        # Evaluate models
        (mse_train, mse_good_eval, mse_bad_eval) = evaluate_models_train(histstruct)
        # Fraction of MSEs to use in WP definition vs testing
        wp_test_split = 0.6
        (wpData, sep) = fit_mse_distribution(mse_good_eval[:int(wp_test_split * len(mse_good_eval))], mse_bad_eval[:int(wp_test_split * len(mse_bad_eval))], wpBiasFactor)
        (f_measure, accuracy, precision, recall) = evaluate_autoencoders_combined(mse_good_eval, mse_bad_eval, wpData, wp_test_split)

        gpu_check()    

        for j, autoencoder in enumerate(autoencoders):
            autoencoder.save('../SavedModels/Permutations/Job' + str(i + 1) + '/AE' + str(j))
        del(autoencoders)
        # Creating a debug file for assessing autoencoder postprocessing
        debug.append([mse_train, mse_good_eval, mse_bad_eval, wpData])
        np.array(debug)
        np.save('Debug', debug)

        # Empty list
        dataPackage = [histnames, i + 1, trainTime, sep, f_measure, accuracy, precision, recall]
        if len(aeStats) < 1:
            aeStats.append(dataPackage)
            print('New Best Model:')
            
        # Non-empty List
        else:
            for j in range(len(aeStats) - 1, -1, -1):
                if f_measure < aeStats[j][4]:
                    aeStats.insert(j+1, dataPackage)
                    print('Model Position: ' + str(j + 1))
                    break
                elif f_measure == aeStats[j][4]:
                    if accuracy < aeStats[j][5]:
                        aeStats.insert(j+1, dataPackage)
                        print('Model Position: ' + str(j + 1))
                        break
                # Reached end of list
                if j == 0:
                    aeStats.insert(j, dataPackage)
                    print('New Best Model:')
        print(' - Train Time: ' + str(trainTime))
        print(' - Separability: ' + str(sep))
        print(' - Accuracy: ' + str(accuracy))
        print(' - F{}-Measure: '.format(fmBiasFactor) + str(f_measure))
                
        print()
    except MemoryError as e:
        print('ERROR: Encountered exception in job ' + str(i+1), file=sys.stderr)
        print('ERROR encountered in job {}. Exiting...'.format(i+1))
        aeStats.append(['ERROR', i + 1, 0, 0.0, 0, 0.0, 0, 0, 0])
        raise MemoryError(e)
    except Exception as e:
        print('ERROR: Encountered exception in job ' + str(i+1), file=sys.stderr)
        print('ERROR encountered in job {}. Continuing...'.format(i+1))
        print(e)
        aeStats.append(['ERROR', i + 1, 0, 0.0, 0, 0.0, 0, 0, 0])
    return aeStats, numModels, debug


# In[ ]:
### Function to prevent exhausting shared resources
def gpu_check():
    # Prevents crashing on CPU only runs
    if len(tf.config.list_physical_devices('GPU')) < 1:
        if (psutil.virtual_memory()[1] / psutil.virtual_memory()[0]) < 0.15:
            raise MemoryError('Excessive RAM Usage!')
        return
    # Get peak memory usage of GPU
    usage = tf.config.experimental.get_memory_info('GPU:0')
    print('Using {} GB of GPU Memory'.format(usage['peak'] / 1000000000.0))
    if usage['peak'] > 7000000000.0:
        raise MemoryError('Excessive GPU Memory Usage!')


def memlocker():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return (info.free < 3000000000), info.free
    
### Main loop to iterate through possible histlists
memlock = True
while(memlock):
    gpuData = memlocker()
    if(gpuData[0]):
        memlock = False
        break
    else:
        sys.stdout.write(str(gpuData/1000000000) + ' GB Available - Waiting for sufficient memory...')
        sys.stdout.flush()
        time.sleep(2)


userfriendly = True
aeStats = []
debug = []
numModels = 0
#sys.sdout = open('HistPerm.log' , 'w')
for i,histnames in enumerate(histlists):
    #tracemalloc.start()
    (aeStats, numModels, debug) = masterLoop(aeStats, numModels, histnames, histstruct, debug)
    #snapshot = tracemalloc.take_snapshot()
    #display_top(snapshot)
    df = pd.DataFrame(aeStats, columns=['Histnames', 'Job', 'Train Time', 'Average Separation', 'F_Measure', 'Accuracy', 'Precision', 'Recall'])
    csvu.write_csv(df, 'Top50.csv')
    gc.collect()
    K.clear_session()

#sys.sdout.close()

#!/usr/bin/env python
# coding: utf-8

# In[97]:

