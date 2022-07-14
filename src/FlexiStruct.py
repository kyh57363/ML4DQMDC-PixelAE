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
    """ A class to extend the HistStruct class and implement a more flexible, workable design for advanced testing. """
    
    
    def __init__( self ):
        """An empty initializer, setting all containers to empty defaults.
        A HistStruct object has the following properties:
        histlist: a list of histograms containing the substructure as a set of sublists (ie. [[hist1, hist2], [hist3, hist4, hist5]]"""
        
        super().__init__()
        histlist = []
        
    def add_dataframe( self, df, cropslices=None, rebinningfactor=None, 
                        smoothinghalfwindow=None, smoothingweights=None,
                        donormalize=True, standardbincount = 0):
        ### add a dataframe to a HistStruct
        # input arguments:
        # - df: a pandas dataframe as read from the input csv files
        # - cropslices: list of slices (one per dimension) by which to crop the histograms
        # - rebinningfactor: factor by which to group bins together
        # - smoothinghalfwindow: half window (int for 1D, tuple for 2D) for doing smoothing of histograms
        # - smoothingweights: weight array (1D for 1D, 2D for 2D) for smoothing of histograms
        # - donormalize: boolean whether to normalize the histograms
        # - standardbincount: creates padded arrays for histograms with different bin counts to standardize for model training. 0 is default, which does not do any padding (normal HistStruct behavior)
        # for more details on cropslices, rebinningfactor, smoothingwindow, smoothingweights
        # and donormalize: see hist_utils.py!
        # notes:
        # - the new dataframe can contain one or multiple histogram types
        # - the new dataframe must contain the same run and lumisection numbers (for each histogram type in it)
        #   as already present in the HistStruct, except if it is the first one to be added
        # - alternative to adding the dataframe with the options cropslices, donormalize and rebinningfactor
        #   (that will be passed down to preparedatafromdf), one can also call preparedatafromdf manually and add it
        #   with add_histograms, allowing for more control over complicated preprocessing.
        
        histnames = dfu.get_histnames(df)
        # loop over all names in the dataframe
        for i, histname in enumerate(histnames):
            if histname in self.histnames:
                print('WARNING: Histogram already in HistStruct. Error checking is disabled, so ensure histograms have same features.')
            thisdf = dfu.select_histnames( df, [histname] )
            # determine statistics (must be done before normalizing)
            nentries = np.array(thisdf['entries'])
            # get physical xmin and xmax
            xmin = thisdf.at[0, 'Xmin']
            xmax = thisdf.at[0, 'Xmax']
            # prepare the data
            (hists_all,runnbs_all,lsnbs_all) = hu.preparedatafromdf(thisdf,returnrunls=True,
                                                cropslices=cropslices,
                                                rebinningfactor=rebinningfactor,
                                                smoothinghalfwindow=smoothinghalfwindow,
                                                smoothingweights=smoothingweights,
                                                donormalize=donormalize)
            
            # Get the length of the histograms and make sure they are all consistent for this type
            histlen = 0
            for i, hist in enumerate(hists_all):
                if i == 0:
                    histlen = len(hist)
                else:
                    if histlen != len(hist):
                        raise Exception('ERROR in HistStruct.add_dataframe: histogram bin counts are not self-consistent!')
            
            # Make sure the standardbincount is valid if it is defined
            if histlen > standardbincount and standardbincount > 0:
                raise Exception('ERROR in HistStruct.add_dataframe: standardbincount must be greater than or equal to largest histogram bin count')
            # Pad any histograms with too few bins
            if histlen < standardbincount:
                newHistList = np.zeros((len(hists_all), standardbincount))
                for i, hist in enumerate(hists_all):
                    for j, value in enumerate(hist):
                        newHistList[i][j] = value
            
                hists_all = newHistList
            
            runnbs_all = runnbs_all.astype(int)
            lsnbs_all = lsnbs_all.astype(int)
            
            # check consistency in run and lumisection numbers
            if len(self.histnames)!=0:
                if( not ( (runnbs_all==self.runnbs).all() and (lsnbs_all==self.lsnbs).all() ) ):
                    raise Exception('ERROR in HistStruct.add_dataframe: dataframe run/lumi numbers are not consistent with current HistStruct!')
            # add everything to the structure
            if histname not in self.histnames:
                self.runnbs = runnbs_all
                self.lsnbs = lsnbs_all
                self.histnames.append(histname)
                self.histograms[histname] = hists_all
                self.nentries[histname] = nentries
                self.histranges[histname] = (xmin,xmax)
            else:
                self.runnbs.append(runnbs_all)
                self.lsnbs.append(lsnbs_all)
                self.histograms[histname] = np.concatenate((self.histograms[histname], hists_all))
                self.nentries[histname].append(nentries)
             
    
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
        """ get the array of scores for a given histogram type, optionally after masking
        # input arguments:
        # - histname: name of the histogram type for which to retrieve the score. 
        #   if None, return a dict matching histnames to arrays of scores
        # - masknames: list of names of masks (default: no masking, return full array)
        # notes:
        # - this method takes the scores from the HistStruct.scores attribute;
        #   make sure to have evaluated the classifiers before calling this method,
        #   else an exception will be thrown."""
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
        """### similar to get_scores, but with different return type:
        # np array of shape (nhistograms, nhistogramtypes)"""
        scores = self.get_scores( masknames=masknames )
        scores_array = []
        for histgroup in self.histlist:
            for hist in histgroup:
                scores_array.append(scores[hist])
        scores_array = np.transpose(np.array(scores_array))
        return scores_array
        