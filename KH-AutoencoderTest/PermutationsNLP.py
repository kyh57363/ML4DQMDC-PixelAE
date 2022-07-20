### imports
# external modules
import os
import gc
from os.path import exists
import os.path
from xxlimited import foo
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

# Select data to be used in training
year = '2017'
eras = ['E', 'F']

### Controls
# Bias Factors
fmBiasFactor = 2
wpBiasFactor = 3
badThreshold = 3

# Set up data directory location
datadir = {}
for era in eras:
    datadir[year+era]  = '../data/' + year + era + '/'

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

histnames = histlists[255]

runsls_training = {}
runsls_bad = {}
runsls_good = {}
for era in eras:
    runsls_training.update(trainrunsls[year + era])
    # Select bad runs to test on in the user-defined list
    runsls_bad.update(badrunsls[year + era])
    # Select good runs to test on in the user-defined list
    runsls_good.update(goodrunsls[year + era])


### Data Import
dloader = DataLoader.DataLoader()
histstruct = FlexiStruct.FlexiStruct()
histstruct.reset_histlist(histnames)
failedruns = {}
failedls ={}
# Unpack histnames and add every histogram individually
consistent = True
sys.stdout = open('HistPerm.log' , 'w')
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
                runsls_total = {k: v for d in (runsls_training, runsls_good, runsls_mse, runsls_bad) for k, v in d.items()}
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
sys.stdout.close()

# Give user information on failed imports
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


##### FUNCTIONS SECTION #####

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

### Defines the autoencoders for training
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

### Trains a given set of autoencoders
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

### Gets MSE for a given group of histograms
def getMSEs(histstruct, histgroup, masknames=None):
    mse_normhist = np.array([hu.normalize(histstruct.get_histograms(
                histname = hname, masknames = masknames), 
                                                 norm="l1", axis=1) 
                                       for hname in histnamegroup]).transpose((1,0,2))
    
    predictions = np.array(histstruct.classifiers[histgroup[0]].model.predict([mse_normhist[:,i] for i in range(mse_normhist.shape[1])]))
    hists_eval = []
    for i in range(len(histgroup)):
        ls_eval = []
        for ls in mse_normhist:
            ls_eval.append(ls[i])
        hists_eval.append(ls_eval)

    # Outputs scores by histogram by lumisection
    scoreslist = []
    for i in range(len(histgroup)):
        data = np.array(hists_eval[i])
        preds = predictions[i]
        sqdiff = np.power(data-preds,2)
        sqdiff[:,::-1].sort()
        sqdiff = sqdiff[:,:10]
        mean = np.mean(sqdiff,axis=-1)
        scoreslist.append(mean)
    
        histstruct.scores[histgroup[i]] = mean

    return scoreslist

### Get Predictions from the Models for WP definition
def predict_models_train(histstruct, histnames):

    # Iterates through every histogram and finds mses for the subsets
    wps = {}
    goodPreds = {}
    badPreds = {}
    mse_trains = []
    mse_goods = []
    mse_bads = []
    for histgroup in histnames:
        mse_train = getMSEs(histstruct, histgroup, masknames=['dcson','highstat', 'training'])
        mse_good = getMSEs(histstruct, histgroup, masknames=['dcson','highstat', 'good'])
        mse_bad = getMSEs(histstruct, histgroup, masknames=['dcson','highstat', 'bad'])
        
        # Save for debugging
        mse_trains.append(mse_train)
        mse_goods.append(mse_good)
        mse_bads.append(mse_bad)

        # Determine whether the model predicts each lumisection good or bad
        for i, histname in histnamegroup:
            wp = np.mean(mse_train[i]) + wpBiasFactor*np.std(mse_train[i])
            wps[histname] = (wp)

            goodPreds[histname] = np.where(mse_good[i] < wp, 0, 1)
            badPreds[histname] = np.where(mse_bad[i] < wp, 0, 1)
        
    return (wps, goodPreds, badPreds, mse_trains, mse_goods, mse_bads)

### Determine how well the model distinguished good/bad data
def evaluate_models(histnames, goodPreds, badPreds):

    # Getting data into format in which histograms can easily vote
    histname = histnames[0][0]
    goodlumiperhist = np.zeros(len(goodPreds[histname]), (len(goodPreds.keys())))
    badlumiperhist = np.zeros(len(badPreds[histname]), (len(badPreds.keys())))
    histcount = 0
    for histgroup in histnames:
        for histname in histgroup:
            for i in range(len(goodPreds[histname])):
                goodlumiperhist[i][histcount] = goodPreds[histname][i]
            for i in range(len(badPreds[histname])):
                badlumiperhist[i][histcount] = badPreds[histname][i]
    
    goodscores = np.zeros(len(goodlumiperhist))
    for i,lumi in enumerate(goodlumiperhist):
        if np.sum(lumi) > badThreshold:
            goodscores[i] = 1
    
    badscores = np.zeros(len(badlumiperhist))
    for i,lumi in enumerate(badlumiperhist):
        if np.sum(lumi) > badThreshold:
            badscores[i] = 1
    
    scores = np.concatenate(goodscores, badscores)
    labels = np.concatenate(np.zeros(len(goodscores)), np.ones(len(badscores)))

    (tp, fp, tn, fn) = get_metrics(scores, labels)

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    f_measure = (1 + fmBiasFactor * fmBiasFactor) * ((precision * recall) / ((fmBiasFactor * fmBiasFactor * precision) + recall)) 

    return accuracy, precision, recall, f_measure

### returns data for calculating metrics
def get_metrics(scores, labels):
   
    nsig = np.sum(labels)
    nback = np.sum(1-labels)

    # get confusion matrix entries
    tp = np.sum(np.where((labels==1) & (scores==1),1,0))/nsig
    fp = np.sum(np.where((labels==0) & (scores==1),1,0))/nback
    tn = np.sum(np.where((labels==0) & (scores==0),1,0))/nback
    fn = np.sum(np.where((labels==1) & (scores==0),1,0))/nsig

    return (tp, fp, tn, fn)

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
    if usage['peak'] > 2000000000.0:
        raise MemoryError('Excessive GPU Memory Usage!')

### Function for main loop operations
def loopable(histstruct, histnames, numModels, aeStats, debug, i):
    try:
        percComp = (numModels/conmodelcount)*100
        print('Running Job {}/'.format(i+1) + str(len(histlists)) + ' - {:.2f}% Complete'.format(percComp))

        # Update histlist to reflect new data
        histstruct.reset_histlist(histnames, suppress=True)
        assignMasks(histstruct, runsls_training, runsls_good, runsls_bad)

        (histslist, vallist, autoencoders, _) = define_concatamash_autoencoder(histstruct)

        # Train autoencoders based on current histlist and record speed
        start = time.perf_counter()

        # Suppress Unhelpful Error Messages
        orig_out = sys.stderr
        sys.stderr = open('trash', 'w')
        autoencoders = train_concatamash_autoencoder(histstruct, histslist, vallist, autoencoders)
        gpu_check()
        sys.stderr = orig_out
        
        # Gets data for status update to user
        numModels += len(autoencoders)

        stop = time.perf_counter()
        trainTime = stop - start

        # Run model prediction
        (wps, goodPreds, badPreds, mse_trains, mse_goods, mse_bads) = predict_models_train(histstruct, histnames)

        (accuracy, precision, recall, f_measure) = evaluate_models(histnames, goodPreds, badPreds)

        debug.append([wps, mse_trains, mse_goods, mse_bads])
        dataPackage = [histnames, i + 1, trainTime, f_measure, accuracy, precision, recall]
        if len(aeStats) < 1:
            aeStats.append(dataPackage)
            print('New Best Model:')
            
        # Non-empty List
        else:
            for j in range(len(aeStats) - 1, -1, -1):
                if f_measure < (aeStats[j][3]):
                    aeStats.insert(j+1, dataPackage)
                    print('Model Position: ' + str(j + 1))
                    break

                elif f_measure == aeStats[j][3]:
                    if accuracy < aeStats[j][4]:
                        aeStats.insert(j+1, dataPackage)
                        print('Model Position: ' + str(j + 1))
                        break
                # Reached end of list
                if j == 0:
                    aeStats.insert(j, dataPackage)
                    print('New Best Model:')
        
        # Prints every time
        print(' - Train Time: ' + str(trainTime))
        print(' - F{}-Measure: '.format(fmBiasFactor) + str(f_measure))
        print(' - Accuracy: ' + str(accuracy))

        return(numModels, aeStats, debug, i)

    # In case someone else is using all the resources, wait and try again
    except tf.errors.ResourceExhaustedError as e:
        i -= 1
        print('Insufficient Resources! Waiting...')
        time.sleep(30)
        return(numModels, aeStats, debug, i)
    
    # If this program is overutilizing memory, kill the process
    except MemoryError as e:
        print('Overutilization detected! Exiting...')
        raise Exception('Excessive Memory Usage!')
    
    # Other random exceptions are inconsequential and can be passed over
    except Exception as e:
        print('ERROR: Encountered exception in job ' + str(i+1), file=sys.stderr)
        print('ERROR encountered in job {}. Continuing...'.format(i+1))
        print(e)
        aeStats.append(['ERROR', i + 1, 0, 0, 0, 0, 0])
        return(numModels, aeStats, debug, i)
        
### Master Loop
aeStats = []
debug = []
numModels = 0
for i in range(histnames):
    # Execute the main loop ops in a function to prevent memory leaks
    (numModels, aeStats, debug, i) = loopable(histstruct, histnames[i], numModels, aeStats, debug, i)
    
    # Save data for later examination
    df = pd.DataFrame(aeStats, columns=['Histlist', 'Job', 'Train Time',
                                   'Separable Percent Good', 'Worst Case Separation',
                                   'F_measure', 'Working Point', 'Separability', 'Separable Percent Bad'])
    csvu.write_csv(df, 'Top50.csv')
    debugAr = np.array(debug)
    np.save('./DebugData/Debug', arr=debugAr, allow_pickle=True)

    # Ensure memory is cleared
    gc.collect()
    K.clear_session()

sys.stdout.close()