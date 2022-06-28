### imports

# external modules
import os
import sys
import pickle
import zipfile
import glob
import shutil
import copy
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib

# local modules
sys.path.append('classifiers')
from HistogramClassifier import HistogramClassifier
sys.path.append('../utils')
import dataframe_utils as dfu
import hist_utils as hu
import json_utils as jsonu
import plot_utils as pu
from autoencoder_utils import mseTop10Raw, mseTop10
import HistStruct
importlib.reload(HistStruct)


class FlexiStruct( HistStruct.HistStruct ):
    """ A class to extend the HistStruct class and implement functionality for a histogram substructure """
    
    
    def __init__( self ):
        """An empty initializer, setting all containers to empty defaults.
        A HistStruct object has the following properties:
        histlist: a list of histograms containing the substructure as a set of sublists (ie. [[hist1, hist2], [hist3, hist4, hist5]]"""
        
        super().__init__()
        histlist = []
    
    def reset_histlist( self, histlist, suppress=False ):
        """Resets the histlist to preserve histograms but change substructure. Note this requires eliminating all stored classifiers and scores
        input arguments:
             histlist: the new histlist to define the histogram substructure (see class docs)
             suppress: Boolean allowing user to suppress warnings"""
        
        self.globalscores = []
        self.classifiers = {}
        self.scores = {}
        self.extscores = {}
        self.extglobalscores = {}
        self.masks = {}
        if not suppress:print('WARNING: Classifiers and masks cleared to preserve consistency')
        
        self.histlist = histlist
        return self.histlist
   
    def evaluate_classifier( self, histgroup, extname=None ):
        """Evaluate a histogram classifier for a given histogram group in the SubHistStruct.
        input arguments:
          - histgroup: A group of 1+ histograms on which the desired autoencoder was trained
          - extname: name of a set of extra histograms (see add_exthistograms)
                    if None, will evaluate the classifer for the main set of histograms
        notes:
         - the result is both returned and stored in the 'scores' attribute"""
        
        ## Watch dogs
        
        if not histgroup in self.histlist:
            raise Exception('ERROR in SubHistStruct.evaluate_classifier: requested histogram list does not exist in the current histlist')
        
        # 0th index used as the "name" of the classifier by convention
        if not histgroup[0] in self.classifiers.keys():
                raise Exception('ERROR in HistStruct.evaluate_classifier: requested to evaluate classifier for {}'.format(histname)
                           +' but no classifier was set for this histogram name.')    
        
        # Case of not using extra histograms beyond the traditional dataset
        if extname is None:
            histograms = []
            for histname in histgroup:
                
                # Kind of redundant in most cases, but makes sure all histograms exist
                if histname not in self.histnames:
                    raise Exception('ERROR in HistStruct.evaluate_classifier: requested histogram name {}'.format(histname)
                                +' but this is not present in the current HistStruct.')
                
            # Convert training data into form usable by the autoencoder
            X_eval = np.array([hu.normalize(self.get_histograms(
                histname = hname, masknames = None), 
                                                 norm="l1", axis=1) 
                                       for hname in histgroup]).transpose((1,0,2))
                
            # Perform evaluation on histograms
            predictions = np.array(self.classifiers[histgroup[0]].model.predict([X_eval[:,i] for i in range(X_eval.shape[1])]))
            
            # Data exists in a format incompatible with MSETop10Raw function and must be adapted
            hists_eval = []
            for i in range(len(histgroup)):
                ls_eval = []
                for ls in X_eval:
                    ls_eval.append(ls[i])
                hists_eval.append(ls_eval)
            
            # Evaluate data for each individual histogram to treat them seperately
            from autoencoder_utils import mseTop10Raw, mseTop10
            scoreslist = []
            for i in range(len(histgroup)):
                data = np.array(hists_eval[i])
                preds = predictions[i]
                                
                scores = mseTop10Raw( data, preds )
                self.scores[histgroup[i]] = scores
                scoreslist.append(scores)
            
            return scoreslist
        
    def get_scores( self, histname=None, masknames=None ):
        ### get the array of scores for a given histogram type, optionally after masking
        # input arguments:
        # - histname: name of the histogram type for which to retrieve the score. 
        #   if None, return a dict matching histnames to arrays of scores
        # - masknames: list of names of masks (default: no masking, return full array)
        # notes:
        # - this method takes the scores from the HistStruct.scores attribute;
        #   make sure to have evaluated the classifiers before calling this method,
        #   else an exception will be thrown.
        histnames = self.histnames[:]
        if histname is not None:
            # check if histname is valid
            if histname not in self.histnames:
                raise Exception('ERROR in HistStruct.get_scores: requested histogram name {}'.format(histname)
                               +' but this is not present in the current HistStruct.')
            if histname not in self.scores.keys():
                raise Exception('ERROR in HistStruct.get_scores: requested histogram name {}'.format(histname)
                               +' but the scores for this histogram type were not yet initialized.')
            histnames = [histname]
        mask = np.ones(len(self.lsnbs)).astype(bool)
        if masknames is not None:
            mask = self.get_combined_mask(masknames)
        res = {}
        for histgroup in self.histlist:
            for hname in histgroup:
                res[hname] = self.scores[hname][mask]
        if histname is None: return res
        return res[histname]
        
    def get_scores_array( self, masknames=None ):
        ### similar to get_scores, but with different return type:
        # np array of shape (nhistograms, nhistogramtypes)
        scores = self.get_scores( masknames=masknames )
        scores_array = []
        for histgroup in self.histlist:
            for hist in histgroup:
                scores_array.append(scores[hist])
        scores_array = np.transpose(np.array(scores_array))
        return scores_array
        