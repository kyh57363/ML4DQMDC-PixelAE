# to do:
# - see to do's in the code
# - add functionality for plotting the contours with the test sets
# - add functionality for plotting single runs/lumisections
# - continue making styling more uniform (e.g. everything in a Frame)

# external modules

print('importing external modules...')
print('  import os'); import os
print('  import sys'); import sys
print('  import pandas as pd'); import pandas as pd
print('  import numpy as np'); import numpy as np
print('  import matplotlib.pyplot as plt'); import matplotlib.pyplot as plt
print('  import pickle'); import pickle
print('  import functools'); import functools
try: 
    print('importing tkinter for python3')
    import tkinter as tk
    import tkinter.filedialog as fldlg
    from tkinter import ttk
except: 
    print('importing tkinter for python2')
    import Tkinter as tk
    import tkFileDialog as fldlg
    from Tkinter import ttk

# local modules

print('importing utils...')
sys.path.append('../utils')
print('  import csv_utils as csvu'); import csv_utils as csvu
print('  import json_utils as jsonu'); import json_utils as jsonu
print('  import dataframe_utils as dfu'); import dataframe_utils as dfu
print('  import hist_utils as hu'); import hist_utils as hu
print('  import autoencoder_utils as aeu'); import autoencoder_utils as aeu
print('  import plot_utils as pu'); import plot_utils as pu
print('  import generate_data_utils as gdu'); import generate_data_utils as gdu
print('  import refruns_utils as rru'); import refruns_utils as rru
print('  import mask_utils as mu'); import mask_utils as mu

print('importing src...')
sys.path.append('../src')
sys.path.append('../src/classifiers')
sys.path.append('../src/cloudfitters')
print('  import HistStruct'); import HistStruct
print('  import HistogramClassifier'); import HistogramClassifier
print('  import AutoEncoder'); import AutoEncoder
print('  import SeminormalFitter'); import SeminormalFitter
print('  import GaussianKdeFitter'); import GaussianKdeFitter
print('  import HyperRectangleFitter'); import HyperRectangleFitter

print('done')

### mappings

# to do: get list of arguments (and default values) per resampling function / fitter class
#        and return it, so that all key/value text box pairs can be replaced by OptionsFrames!

def get_resampling_function( key=None ):

    allowed = ['None','upsample_hist_set']
    if key is None: return allowed

    key = key.strip(' \t\n')
    if key=='None': return None
    if key=='upsample_hist_set': return gdu.upsample_hist_set
    else:
        raise Exception('ERROR: resampling function {} not recognized.'.format(key))

def get_fitter_class( key=None ):
    
    allowed = ['GaussianKdeFitter']
    if key is None: return allowed

    key = key.strip(' \t\n')
    if key=='GaussianKdeFitter': return GaussianKdeFitter.GaussianKdeFitter
    else:
        raise Exception('ERROR: fitter class {} not recognized'.format(key))

### help functions

def is_float(s):
    try: float(s); return True
    except ValueError: return False

def is_int(s):
    try: int(s); return True
    except ValueError: return False

def is_bool(s):
    if( s=='True' or s=='true' or s=='False' or s=='false' ): return True
    else: return False

def to_bool(s):
    # note: do not use builtin bool() since it appears to return True for every not-None variable
    return (s=='True' or s=='true')

def make_options_dict( keystr, valstr ):
    keys = [key.strip(' ').strip('\t') for key in keystr.split('\n')]
    values = [value.strip(' ').strip('\t') for value in valstr.split('\n')]
    res = {}
    for key, value in zip(keys, values):
        if len(key)==0: continue
        if is_int(value): value = int(value)
        elif is_float(value): value = float(value)
        elif is_bool(value): value = to_bool(value)
        res[key] = value
    return res

def set_frame_default_style( frame ):
    ### apply some default stylings to a tk.Frame
    frame['padx'] = 10
    frame['pady'] = 10
    frame['borderwidth'] = 2
    frame['relief'] = 'groove'

### help classes

class StdOutRedirector:
    ### helper class to redirect print output to GUI widget
    # use as follows:
    #   stdout = sys.stdout
    #   sys.stdout = StdOutRedirector(<some widget>)
    #   ... <some code execution containing print statements>
    #   sys.stdout = stdout

    def __init__(self, tk_text_widget, tk_root_object):
        self.text_dump = tk_text_widget
        self.root = tk_root_object

    def write(self, text):
        self.text_dump.insert(tk.INSERT, text)
        self.text_dump.see(tk.END)
        self.root.update()

    def flush(self):
        # (empty) flush attribute needed to avoid exception on destroying the window
        pass

class OptionsFrame:
    ### contains a tk.Frame holding a list of customization options

    def __init__(self, master, labels=None, types=None, values=None):
        # input arguments:
        # - labels: list of strings with the names/labels of the options
        # - types: list of tk types, defaults to tk.Text for each option
        # - values: list of options passed to each widget
        #           (for now only tk.Text, in wich case values is a list of default values,
        #            but can be extended to e.g. combobox, 
        #            where values would be the options to choose from)
        # note: individual elements of types and values can also be None,
        #       in which case these elements will be set to default
        # to do: extend to other widget types than tk.Text (especially combobox)
        self.frame = tk.Frame(master,width=200)
        self.labels = []
        self.wtypes = []
        self.widgets = []
        if labels is None: 
            raise Exception('ERROR in OptionsFrame initialization:'
                            +' argument "labels" must be specified.')
        if types is None: types = [tk.Text]*len(labels)
        if values is None: values = [None]*len(labels)
        if( len(types)!=len(labels) or len(values)!=len(labels) ):
            raise Exception('ERROR in OptionsFrame initialization:'
                            +' argument lists have unequal lengths.')
            # (to extend error checking)
        for i, (label, wtype, value) in enumerate(zip(labels, types, values)):
            tklabel = tk.Label(self.frame, text=label)
            tklabel.grid(row=i, column=0)
            self.labels.append(tklabel)
            if wtype is None: wtype = tk.Text
            widget = None
            if wtype is tk.Text:
                widget = tk.Text(self.frame, height=1, width=25)
                if value is not None:
                    widget.insert(tk.INSERT, value)
            else:
                raise Exception('ERROR in OptionsFrame initialization:'
                                +' widget type {} not recognized'.format(wtype))
            widget.grid(row=i, column=1)
            self.widgets.append(widget)
            self.wtypes.append(wtype)

    def get_dict(self):
        ### get the options of the current OptionsFrame as a dictionary
        res = {}
        for label, wtype, widget in zip(self.labels, self.wtypes, self.widgets):
            key = label.cget('text')
            value = None
            if wtype is tk.Text:
                value = widget.get('1.0', tk.END)
            else:
                raise Exception('ERROR in OptionsFrame get_dict:'
                               +' no getter method implemented for widget type {}'.format(wtype))
            value = value.strip(' \t\n')
            if is_int(value): value = int(value)
            elif is_float(value): value = float(value)
            elif is_bool(value): value = to_bool(value)
            res[key] = value
        return res

class ScrolledFrame:
    ### contains a tk.Frame holding a text widget with vertical and horizontal scrollbars
    # note: it does not seem possible to just pass an arbitrary widget in the constructor,
    #       since the widget must have its master (i.e. this frame) set on creation.
    #       therefore, first create the ScrolledFrame f, then the widget (using the f as master),
    #       and then call set_widget to position the widgt correctly in the frame.
    #       to do: find out if this can be simplified...

    def __init__(self, master, height=50, width=50, adaptable_size=False):
        # note: if adaptable_size is True, the Frame will take its size from the child widget
        self.frame = tk.Frame(master, height=height, width=width)
        if not adaptable_size: self.frame.grid_propagate(0)

    def set_widget(self, widget):
        self.widget = widget
        widget.grid(row=0, column=0, sticky='nsew')
        self.yscrollbar = tk.Scrollbar(self.frame, orient="vertical", command=widget.yview)
        self.xscrollbar = tk.Scrollbar(self.frame, orient="horizontal", command=widget.xview)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

class ScrolledTextFrame(ScrolledFrame):
    ### specific case of ScrolledFrame, where the widget is tk.Text.
    # note: the advantage of using this specification is that the tk.Text widget 
    #       does not need to be created manually;
    #       it is created internally and accessible via the .widget attribute.

    def __init__(self, master, txtheight=50, txtwidth=50):
        super().__init__(master, adaptable_size=True)
        text = tk.Text(self.frame, height=txtheight, width=txtwidth)
        self.set_widget(text)

### GUI windows

class PlotSetsWindow(tk.Toplevel):
    ### popup window class for plotting the histograms in a histstruct

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Plotting')
        self.histstruct = histstruct
        self.set_optionsframe_list = []
        self.select_set_button_list = []
        self.set_selector_list = []

        # make a frame holding the action buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style(self.buttons_frame)

        # add a button to allow adding more sets of options
        self.more_button = tk.Button(self.buttons_frame, text='Add...', command=self.add_set)
        self.more_button.grid(row=0, column=0)

        # add a button to make the plot
        self.plot_button = tk.Button(self.buttons_frame, text='Plot', command=self.make_plot)
        self.plot_button.grid(row=0, column=1)

        # add one set of options
        self.add_set()

    def add_set(self):
        ### add widgets for one more histogram set to plot
        column = 0
        idx = len(self.set_selector_list)
        row = 1+len(self.set_selector_list)
        set_frame = tk.Frame(self)
        set_frame.grid(row=row, column=column, sticky='nsew')
        set_frame_default_style(set_frame)
        select_set_button = tk.Button(set_frame, text='Select set',
                                    command=functools.partial(self.open_selection_window,idx),
                                    bg='orange')
        select_set_button.grid(row=0, column=0)
        self.select_set_button_list.append(select_set_button)
        set_default_options = {'label':None, 'color':None}
        set_options_frame = OptionsFrame(set_frame,
                                            labels=set_default_options.keys(),
                                            values=set_default_options.values())
        set_options_frame.frame.grid(row=1, column=0)
        self.set_optionsframe_list.append(set_options_frame)
        self.set_selector_list.append( None )

    def open_selection_window(self, idx):
        self.set_selector_list[idx] = SelectorWindow(self.master, self.histstruct,
                                                        only_mask_selection=True)
        self.select_set_button_list[idx]['bg'] = 'green'
        # (to do: make check if sets were actually selected more robust, as selection window
        #         could be closed without making a selection, but button would still be green.)

    def check_all_selected(self):
        if None in self.set_selector_list: return False
        else: return True

    def make_plot(self):
        ### make the plot with current settings
        if not self.check_all_selected():
            raise Exception('ERROR: some sets were declared but not initialized.')
        optionsdict = {'masknames':[], 'labellist':[], 'colorlist':[]}
        for optionsframe, setselector in zip(self.set_optionsframe_list,self.set_selector_list):
            masks = setselector.masks
            optionsdict['masknames'].append( masks )
            setoptions = optionsframe.get_dict()
            optionsdict['labellist'].append( setoptions['label'] )
            optionsdict['colorlist'].append( setoptions['color'] )
        print('found following plotting options:')
        print(optionsdict)
        print('making plot...')
        fig,axs = self.histstruct.plot_histograms( **optionsdict )
        # save or plot the figure
        #fig.savefig('test.png')
        plt.show(block=False)
        # close the window
        self.destroy()
        self.update()
        print('done')

class DisplayHistStructWindow(tk.Toplevel):
    ### popup window class for displaying full info on a HistStruct

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('HistStruct display')
        self.histstruct = histstruct

        # add scrollview widget for displaying large text
        self.viewframe = ScrolledTextFrame(self, txtheight=30, txtwidth=90)
        self.viewframe.widget.insert(tk.INSERT, self.get_info_str())
        self.viewframe.frame.grid(row=0, column=0)

    def get_info_str(self):
        return self.histstruct.__str__()

class SelectorWindow(tk.Toplevel):
    ### popup window class for histogram selection
    # (common to several other popup windows)

    def __init__(self, master, histstruct, only_mask_selection=False,
                                            only_set_selection=False,
                                            allow_multi_mask=True,
                                            allow_multi_set=False):
        super().__init__(master=master)
        self.title('Histogram selection')
        self.histstruct = histstruct
        self.histograms = None
        self.masks = None
        self.sets = None
        self.scores = None
        self.globalscores = None

        # add widgets for choosing masks
        self.histstruct_masks_label = tk.Label(self, text='Choose masks')
        if not only_set_selection: self.histstruct_masks_label.grid(row=0,column=0)
        mask_selectmode = 'multiple' if allow_multi_mask else 'single'
        self.histstruct_masks_listbox = tk.Listbox(self, selectmode=mask_selectmode,
                                                            exportselection=False)
        for maskname in self.histstruct.get_masknames():
            self.histstruct_masks_listbox.insert(tk.END, maskname)
        if len(self.histstruct.get_masknames())==0:
            self.histstruct_masks_listbox.insert(tk.END, '[no masks available]')
        if not only_set_selection: self.histstruct_masks_listbox.grid(row=1,column=0)
        
        # add widgets for choosing a (resampled) set directly
        self.histstruct_sets_label = tk.Label(self, text='Choose sets')
        if not only_mask_selection: self.histstruct_sets_label.grid(row=2,column=0)
        set_selectmode = 'multiple' if allow_multi_set else 'single'
        self.histstruct_sets_listbox = tk.Listbox(self, selectmode=set_selectmode,
                                                        exportselection=False)
        for extname in self.histstruct.exthistograms.keys():
            self.histstruct_sets_listbox.insert(tk.END, extname)
        if len(self.histstruct.exthistograms.keys())==0:
            self.histstruct_sets_listbox.insert(tk.END, '[no sets available]')
        if not only_mask_selection: self.histstruct_sets_listbox.grid(row=3,column=0)

        # add widget for selection
        self.select_button = tk.Button(self, text='Select', command=self.select_histograms)
        self.select_button.grid(row=4, column=0)

    def get_masks(self):
        ### get currently selected masks
        # warning: do not use after selection window has been closed.
        masks = ([self.histstruct_masks_listbox.get(idx)
                    for idx in self.histstruct_masks_listbox.curselection() ])
        return masks

    def get_sets(self):
        ### get currently selected sets
        # warning: do not use after selection window has been closed.
        sets = ([self.histstruct_sets_listbox.get(idx)
                    for idx in self.histstruct_sets_listbox.curselection() ])
        return sets

    def get_histograms(self):
        ### get currently selected histograms
        return self.histograms

    def get_scores(self):
        ### get scores of currently selected histograms
        if self.scores is None:
            print('WARNING: the current histogram selection does not contain scores.'
                    +' Did you properly evaluate a model on the selected set?')
        return self.scores

    def get_globalscores(self):
        ### get global scores of currently selected lumisections
        if self.globalscores is None:
            print('WARNING: the current lumisection selection does not contain global scores.'
                    +' Did you properly evaluate a model on the selected set?')
        return self.globalscores

    def select_histograms(self):
        ### set the histograms based on the current user settings
        masks = self.get_masks()
        do_masks = bool(len(masks)>0)
        sets = self.get_sets()
        do_sets = bool(len(sets)>0)
        if( not do_masks and not do_sets ):
            raise Exception('ERROR: you must select either at least one mask or a training set.')
        if( do_masks and do_sets ):
            raise Exception('ERROR: you cannot select both masks and sets.')
        if do_masks:
            self.histograms = self.histstruct.get_histograms(masknames=masks)
            self.masks = masks
            if len(self.histstruct.scores.keys())>0:
                self.scores = self.histstruct.get_scores(masknames=masks)
            if len(self.histstruct.globalscores)>0:
                self.globalscores = self.histstruct.get_globalscores(masknames=masks)
        else:
            extname = sets[0]
            self.histograms = self.histstruct.get_exthistograms(extname)
            self.sets = sets
            if( extname in self.histstruct.extscores.keys()
                and len(self.histstruct.extscores[extname].keys())>0 ):
                self.scores = self.histstruct.get_extscores(extname)
            if extname in self.histstruct.extglobalscores.keys():
                self.globalscores = self.histstruct.get_extglobalscores(extname)
        # close the window
        self.destroy()
        self.update()
        print('done')

class TrainClassifiersWindow(tk.Toplevel):
    ### popup window class for adding classifiers

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Training')
        self.histstruct = histstruct
        self.training_set_selector = None

        # create frame for options
        self.train_options_frame = tk.Frame(self)
        self.train_options_frame.grid(row=0, column=0)
        set_frame_default_style( self.train_options_frame )
        self.train_options_label = tk.Label(self.train_options_frame, text='Training settings')
        self.train_options_label.grid(row=0, column=0, columnspan=2)

        # add widget to select histograms
        self.select_train_button = tk.Button(self.train_options_frame, 
                                            text='Select training set',
                                            command=self.open_training_selection_window,
                                            bg='orange')
        self.select_train_button.grid(row=1, column=0, columnspan=2)

        # add widgets for training options
        self.key_label = tk.Label(self.train_options_frame, text='Keys')
        self.key_label.grid(row=2, column=0)
        self.key_text = ScrolledTextFrame(self.train_options_frame, txtheight=5, txtwidth=25)
        self.key_text.frame.grid(row=3, column=0)
        self.value_label = tk.Label(self.train_options_frame, text='Values')
        self.value_label.grid(row=2, column=1)
        self.value_text = ScrolledTextFrame(self.train_options_frame, txtheight=5, txtwidth=25)
        self.value_text.frame.grid(row=3, column=1)

        # add button to start training
        self.train_button = tk.Button(self, text='Start training', command=self.do_training)
        self.train_button.grid(row=4, column=0, columnspan=2)

    def open_training_selection_window(self):
        self.training_set_selector = SelectorWindow(self.master, self.histstruct)
        self.select_train_button['bg'] = 'green'
        return

    def get_training_options(self):
        ### get current options entered by the user
        keystr = self.key_text.widget.get(1.0,tk.END)
        valstr = self.value_text.widget.get(1.0,tk.END)
        return make_options_dict( keystr, valstr )

    def do_training(self):
        training_options = self.get_training_options()
        if self.training_set_selector is None:
            raise Exception('ERROR: please select a training set before starting training.')
        training_histograms = self.training_set_selector.get_histograms()
        for histname in training_histograms.keys():
            # check if a classifier is initialized for this histogram type
            if histname not in self.histstruct.classifiers.keys():
                print('WARNING: no classifier was found in the HistStruct'
                        +' for histogram type {}; skipping.'.format(histname))
                continue
            hists = training_histograms[histname]
            print('training a classifier for {}'.format(histname))
            print('size of training set: {}'.format(hists.shape))
            # do training
            self.histstruct.classifiers[histname].train( hists, **training_options )
            # do evaluation
            print('evaluating model for '+histname)
            self.histstruct.evaluate_classifier( histname )
        # close the window
        self.destroy()
        self.update()
        print('done')


class FitWindow(tk.Toplevel):
    ### popup window class for fitting classifier outputs

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Fitting')
        self.histstruct = histstruct
        self.fitting_set_selector = None

        # create frame for options
        self.fit_options_frame = tk.Frame(self)
        self.fit_options_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.fit_options_frame )
        self.fit_options_label = tk.Label(self.fit_options_frame, text='Fit settings')
        self.fit_options_label.grid(row=0, column=0, columnspan=2)

        # add widgets for choosing fitting set
        self.select_fitting_set_button = tk.Button(self.fit_options_frame, 
                                                text='Select fitting set',
                                                command=self.open_fitting_set_selection_window,
                                                bg='orange')
        self.select_fitting_set_button.grid(row=1, column=0, columnspan=2)

        # add widgets to select fitting parameters
        self.fitter_label = tk.Label(self.fit_options_frame, text='Fitter type')
        self.fitter_label.grid(row=2,column=0)
        self.fitter_box = ttk.Combobox(self.fit_options_frame, values=get_fitter_class())
        self.fitter_box.current(0)
        self.fitter_box.grid(row=2,column=1)
        self.fitter_keys_label = tk.Label(self.fit_options_frame, text='Keys')
        self.fitter_keys_label.grid(row=3,column=0)
        self.fitter_keys_text = ScrolledTextFrame(self.fit_options_frame, 
                                                    txtheight=5, txtwidth=15)
        self.fitter_keys_text.frame.grid(row=4,column=0)
        self.fitter_values_label = tk.Label(self.fit_options_frame, text='Values')
        self.fitter_values_label.grid(row=3,column=1)
        self.fitter_values_text = ScrolledTextFrame(self.fit_options_frame, 
                                                    txtheight=5, txtwidth=15)
        self.fitter_values_text.frame.grid(row=4,column=1)

        # create frame for plotting options
        self.plot_options_frame = tk.Frame(self)
        self.plot_options_frame.grid(row=1, column=0, sticky='nsew')
        set_frame_default_style( self.plot_options_frame )
        self.plot_options_label = tk.Label(self.plot_options_frame, text='Plot settings')
        self.plot_options_label.grid(row=0, column=0)
    
        # add widgets for plotting options
        plot_options_dict = ({
                         # options needed at this level:
                         'do_plot':'True',
                         # options passed to plot function:
                         'logprob':'True', 'clipprob':'True',
                         'onlycontour':'False', 'xlims':'30', 'ylims':'30',
                         'onlypositive':'True', 'transparency':'0.5',
                         'title':'density fit of lumisection MSE'})
        self.plot_options = OptionsFrame(self.plot_options_frame,
                                            labels=plot_options_dict.keys(),
                                            values=plot_options_dict.values())
        self.plot_options.frame.grid(row=1, column=0)

        # add widgets to start the fit
        self.fit_button = tk.Button(self, text='Start fit', command=self.do_fit)
        self.fit_button.grid(row=2, column=0, columnspan=2)

    def open_fitting_set_selection_window(self):
        self.fitting_set_selector = SelectorWindow(self.master, self.histstruct)
        self.select_fitting_set_button['bg'] = 'green'
        return
    
    def get_fitting_scores(self):
        if self.fitting_set_selector is None:
            raise Exception('ERROR: please select a set to fit to before doing the fit.')
        scores_fit_dict = self.fitting_set_selector.get_scores()
        if scores_fit_dict is None:
            raise Exception('ERROR: no valid scores could be found in the HistStruct '
                            +'for the specified fitting set.')
        scores_fit = []
        for histname in self.histstruct.histnames:
            thisscore = scores_fit_dict[histname]
            scores_fit.append( thisscore )
        # transform to arrays with correct shape
        scores_fit = np.array(scores_fit)
        scores_fit = np.transpose(scores_fit)
        print('found score array for fitting set of following shape: {}'.format(scores_fit.shape))
        return scores_fit

    def get_fitter(self):
        fitter_name = self.fitter_box.get()
        fitter = get_fitter_class( fitter_name )
        fitter_keystr = self.fitter_keys_text.widget.get(1.0,tk.END)
        fitter_valstr = self.fitter_values_text.widget.get(1.0,tk.END)
        fitter_options = make_options_dict(fitter_keystr, fitter_valstr)
        return (fitter,fitter_options)

    def do_fit(self):
        fitting_scores = self.get_fitting_scores()
        (fitter,fitter_options) = self.get_fitter()
        plot_options_dict = self.plot_options.get_dict()
        do_plot = plot_options_dict.pop('do_plot')
        if do_plot:
            dimslist = []
            fitfunclist = []
            nhisttypes = len(self.histstruct.histnames)
            for i in range(0,nhisttypes-1):
                for j in range(i+1,nhisttypes):
                    dimslist.append((i,j))
            plt.close('all')
            for dims in dimslist:
                thismse = fitting_scores[:,dims]
                fitfunc = fitter( thismse, **fitter_options )
                (fig,ax) = pu.plot_fit_2d(thismse, fitfunc=fitfunc,
                                    xaxtitle=self.histstruct.histnames[dims[0]],
                                    yaxtitle=self.histstruct.histnames[dims[1]],
                                    **plot_options_dict)
                fitfunclist.append(fitfunc)
                plt.show(block=False)
        self.histstruct.fitfunc = fitter( fitting_scores, **fitter_options )
        # to do: extend HistStruct class to contain the fitfunc in a cleaner way!
        #        (or decide on another way to make this ad-hod attribute assignment more clean)
        # evaluate the fitted function on the non-extended histstruct
        scores_all = []
        for histname in self.histstruct.histnames:
            thisscore = self.histstruct.get_scores( histname=histname )
            scores_all.append( thisscore )
        scores_all = np.array(scores_all)
        scores_all = np.transpose(scores_all)
        self.histstruct.add_globalscores( np.log(self.histstruct.fitfunc.pdf(scores_all)) )
        # close the window
        self.destroy()
        self.update()
        print('done')


class ResampleWindow(tk.Toplevel):
    ### popup window class for resampling testing and training sets

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Resampling window')
        self.histstruct = histstruct
        self.set_widget_list = []
        self.set_selector_list = []

        # create a frame for the buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0, sticky='nsew')
        set_frame_default_style( self.buttons_frame )
        
        # add widget to add a set
        self.add_button = tk.Button(self.buttons_frame, text='Add', command=self.add_set)
        self.add_button.grid(row=1, column=0)

        # add widgets to start resampling
        self.resample_button = tk.Button(self.buttons_frame, text='Start resampling',
                                        command=self.do_resampling)
        self.resample_button.grid(row=2,column=0)

        # add widgets to view current resampled sets
        self.view_frame = tk.Frame(self)
        self.view_frame.grid(row=0, column=1)
        set_frame_default_style( self.view_frame )
        self.sets_label = tk.Label(self.view_frame, text='Existing sets')
        self.sets_label.grid(row=0, column=0)
        self.sets_listbox = tk.Listbox(self.view_frame, selectmode='multiple',
                                    exportselection=False)
        self.update_sets_list()
        self.sets_listbox.grid(row=1,column=0)

        # add one set
        self.add_set()

    def update_sets_list(self):
        self.sets_listbox.delete(0, tk.END)
        extnames = self.histstruct.exthistograms.keys()
        for extname in extnames:
            self.sets_listbox.insert(tk.END, extname)
        if len(extnames)==0:
            self.sets_listbox.insert(tk.END, '[no sets available]')

    def add_set(self):
        ### add widgets for one more histogram set to plot
        column = len(self.set_widget_list)+2
        row = 0
        idx = len(self.set_widget_list)
        
        # create a frame to hold the widgets
        set_frame = tk.Frame(self)
        set_frame.grid(row=row, column=column)
        set_frame_default_style( set_frame )

        # add widgets for choosing resampling basis set
        select_button = tk.Button(set_frame, text='Select set',
                                    command=functools.partial(self.open_select_window,idx),
                                    bg='orange')
        select_button.grid(row=0, column=0, columnspan=2)

        # add widgets for resampling options
        name_label = tk.Label(set_frame, text='Name')
        name_label.grid(row=1, column=0)
        name_text = tk.Text(set_frame, height=1, width=15)
        name_text.grid(row=1, column=1)
        partitions_label = tk.Label(set_frame, text='Partitions')
        partitions_label.grid(row=2, column=0)
        partitions_text = tk.Text(set_frame, height=1, width=15)
        partitions_text.insert(tk.INSERT, '-1')
        partitions_text.grid(row=2, column=1)
        function_label = tk.Label(set_frame, text='Function')
        function_label.grid(row=3, column=0)
        function_box = ttk.Combobox(set_frame, values=get_resampling_function())
        function_box.current(0)
        function_box.grid(row=3, column=1)
        function_keys_label = tk.Label(set_frame, text='Keys')
        function_keys_label.grid(row=4, column=0)
        function_keys_text = tk.Text(set_frame, height=5, width=15)
        function_keys_text.grid(row=5, column=0)
        function_values_label = tk.Label(set_frame, text='Values')
        function_values_label.grid(row=4, column=1)
        function_values_text = tk.Text(set_frame, height=5, width=15)
        function_values_text.grid(row=5, column=1)

        self.set_widget_list.append({'select_button':select_button,
                                    'name':name_text,'partitions':partitions_text,
                                    'function':function_box,'function_keys':function_keys_text,
                                    'function_values':function_values_text})
        self.set_selector_list.append( None )


    def open_select_window(self, idx):
        self.set_selector_list[idx] = SelectorWindow(self.master, self.histstruct)
        self.set_widget_list[idx]['select_button']['bg'] = 'green'
        return

    def check_all_selected(self):
        if None in self.test_set_selector_list: return False
        else: return True

    def get_name(self, setindex):
        text = self.set_widget_list[setindex]['name']
        nstr = text.get(1.0,tk.END)
        name = nstr.strip(' \t\n')
        return name 

    def get_partitions(self, setindex):
        ### get currently entered partitions
        text = self.set_widget_list[setindex]['partitions']
        pstr = text.get(1.0,tk.END)
        return int(pstr)

    def get_function(self, setindex):
        widgets = self.set_widget_list[setindex]
        function_name = widgets['function'].get()
        function = get_resampling_function( function_name )
        function_keystr = widgets['function_keys'].get(1.0,tk.END)
        function_valstr = widgets['function_values'].get(1.0,tk.END)
        function_options = make_options_dict(function_keystr, function_valstr)
        return (function, function_options)

    def do_resampling(self):

        nsets = len(self.set_widget_list)
        for i in range(nsets):
            print('generating data for set {} of {}'.format(i+1,nsets))
            hists = self.set_selector_list[i].get_histograms()
            extname = self.get_name(i)
            partitions = self.get_partitions(i)
            (function, function_options) = self.get_function(i)
            for histname in self.histstruct.histnames:
                print('  now processing histogram type {}'.format(histname))
                thishists = hu.averagehists( hists[histname], partitions )
                exthists = function( thishists, **function_options )
                self.histstruct.add_exthistograms( extname, histname, exthists )
                print('  -> generated {} histograms'.format(len(exthists)))
        self.update_sets_list()
        plt.show(block=False)
        # close the window
        self.destroy()
        self.update()
        print('done')


class EvaluateWindow(tk.Toplevel):
    ### popup window class for evaluating a given model

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Evaluation window')
        self.histstruct = histstruct
        self.select_test_set_button_list = []
        self.test_set_type_box_list = []
        self.test_set_selector_list = []

        # add a button to add more test sets
        self.add_button = tk.Button(self, text='Add', command=self.add_set)
        self.add_button.grid(row=0, column=0)

        # add one test set
        self.add_set()

        # add widgets for score distribution
        self.score_dist_options_label = tk.Label(self, text='Options for score plot')
        self.score_dist_options_label.grid(row=3, column=0)
        score_dist_default_options = ({'siglabel':'anomalous', 'sigcolor':'r', 
                   'bcklabel':'good', 'bckcolor':'g', 
                   'nbins':'200', 'normalize':'True',
                   'xaxtitle':'negative logarithmic probability',
                   'yaxtitle':'number of lumisections (normalized)'})
        self.score_dist_options_frame = OptionsFrame(self, 
                                            labels=score_dist_default_options.keys(),
                                            values=score_dist_default_options.values())
        self.score_dist_options_frame.frame.grid(row=4, column=0)

        # add widgets for roc curve
        self.roc_options_label = tk.Label(self, text='Options for ROC curve')
        self.roc_options_label.grid(row=3, column=1)
        roc_default_options = ({'mode':'geom', 'doprint':'False'})
        self.roc_options_frame = OptionsFrame(self,
                                            labels=roc_default_options.keys(),
                                            values=roc_default_options.values())
        self.roc_options_frame.frame.grid(row=4, column=1)

        # add widgets for confusion matrix
        self.cm_options_label = tk.Label(self, text='Options for confusion matrix')
        self.cm_options_label.grid(row=3, column=2)
        cm_default_options = ({'wp':'50'})
        self.cm_options_frame = OptionsFrame(self,
                                            labels=cm_default_options.keys(),
                                            values=cm_default_options.values())
        self.cm_options_frame.frame.grid(row=4, column=2)

        # add a button to start the evaluation
        self.evaluate_button = tk.Button(self, text='Evaluate', command=self.evaluate)
        self.evaluate_button.grid(row=5, column=0)

    def add_set(self):
        ### add one test set
        row = 1
        column = len(self.select_test_set_button_list)+1
        idx = len(self.select_test_set_button_list)
        select_button = tk.Button(self, text='Select test set', 
                                    command=functools.partial(self.open_select_window,idx),
                                    bg='orange')
        select_button.grid(row=row, column=column)
        self.select_test_set_button_list.append( select_button )
        type_box = ttk.Combobox(self, values=['Good','Bad'])
        type_box.current(0)
        type_box.grid(row=row+1,column=column)
        self.test_set_type_box_list.append( type_box )
        self.test_set_selector_list.append( None )

    def open_select_window(self, idx):
        self.test_set_selector_list[idx] = SelectorWindow(self.master, self.histstruct)
        self.select_test_set_button_list[idx]['bg'] = 'green'
        return

    def check_all_selected(self):
        if None in self.test_set_selector_list: return False
        else: return True

    def get_globalscores(self, test_set_type):
        globalscores = []
        for i in range(len(self.test_set_selector_list)):
            if self.test_set_type_box_list[i].get()!=test_set_type: continue
            globalscores.append(self.test_set_selector_list[i].get_globalscores())
        if len(globalscores)==0:
            print('WARNING: there are no test sets with label {}'.format(test_set_type))
        return globalscores

    def evaluate(self):
        if not self.check_all_selected():
            raise Exception('ERROR: some test sets were declared but not defined')
        globalscores_good_parts = self.get_globalscores('Good')
        globalscores_good = np.concatenate(tuple(globalscores_good_parts))
        globalscores_bad_parts = self.get_globalscores('Bad')
        globalscores_bad = np.concatenate(tuple(globalscores_bad_parts))
        labels_good = np.zeros(len(globalscores_good)) # background: label = 0
        labels_bad = np.ones(len(globalscores_bad)) # signal: label = 1

        labels = np.concatenate(tuple([labels_good,labels_bad]))
        scores = np.concatenate(tuple([-globalscores_good,-globalscores_bad]))
        scores = aeu.clip_scores( scores )

        pu.plot_score_dist(scores, labels, **self.score_dist_options_frame.get_dict())
        auc = aeu.get_roc(scores, labels, **self.roc_options_frame.get_dict())
        aeu.get_confusion_matrix(scores,labels, **self.cm_options_frame.get_dict())
        plt.show(block=False)

class ApplyClassifiersWindow(tk.Toplevel):
    ### popup window class for evaluating the classifiers 

    def __init__(self, master, histstruct):
        super().__init__(master=master)
        self.title('Classifier evaluation')
        self.histstruct = histstruct
        self.set_selector = None
        
        # create a frame for the buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.grid(row=0, column=0)
        set_frame_default_style( self.buttons_frame )
        
        # add a button to select the sets
        self.select_set_button = tk.Button(self.buttons_frame, text='Select sets',
                                            command=self.open_selection_window)
        self.select_set_button.grid(row=0, column=0)

        # add a button to start the evaluation
        self.start_evaluation_button = tk.Button(self.buttons_frame, text='Start evaluation',
                                            command=self.evaluate)
        self.start_evaluation_button.grid(row=1, column=0)

    def open_selection_window(self):
        self.set_selector = SelectorWindow(self.master, self.histstruct, 
                                only_set_selection=True,
                                allow_multi_set=True)

    def evaluate(self):
        extnames = self.set_selector.sets
        for extname in extnames:
            print('evaluating classifiers on set {}'.format(extname))
            for histname in self.histstruct.histnames:
                print('  now processing histogram type {}'.format(histname))
                self.histstruct.evaluate_classifier( histname, extname=extname )
        # close the window
        self.destroy()
        self.update()
        print('done')

class ApplyFitWindow(ApplyClassifiersWindow):
    ### popup window class for evaluating the fitter

    def __init__(self, master, histstruct):
        super().__init__(master, histstruct)
        self.title('Fitter evaluation')

    def evaluate(self):
        extnames = self.set_selector.sets
        for extname in extnames:
            print('evaluating fitter on set {}'.format(extname))
            scores_all = []
            for histname in self.histstruct.histnames:
                scores_all.append( self.histstruct.get_extscores( extname, histname=histname ) )
            scores_all = np.array(scores_all)
            scores_all = np.transpose(scores_all)
            self.histstruct.add_extglobalscores( extname, 
                            np.log(self.histstruct.fitfunc.pdf(scores_all)) )
        # close the window
        self.destroy()
        self.update()
        print('done')


class ML4DQMGUI:

    def __init__(self, master):
        
        self.master = master
        master.title('ML4DQM GUI')

        # initializations
        self.histstruct = None
        self.histstruct_filename = None
        self.button_frames = []
        self.all_frames = []

        # add widgets for loading and saving a HistStruct
        self.iobutton_frame = tk.Frame(master)
        self.iobutton_label = tk.Label(self.iobutton_frame, text='HistStruct I/O')
        self.iobutton_label.grid(row=0, column=0)
        self.load_button = tk.Button(self.iobutton_frame, text='Load',
                                     command=self.load_histstruct)
        self.load_button.grid(row=1, column=0)
        self.save_button = tk.Button(self.iobutton_frame, text='Save',
                                     command=self.save_histstruct)
        self.save_button.grid(row=2, column=0)
        self.display_histstruct_button = tk.Button(self.iobutton_frame, 
                                                    text='Display',
                                                    command=self.open_display_histstruct_window)
        self.display_histstruct_button.grid(row=3, column=0)
        self.update_histstruct_info_button = tk.Button(self.iobutton_frame, text='Refresh',
                                                command=self.update_histstruct_info)
        self.update_histstruct_info_button.grid(row=4, column=0)
        # add the frame to the window
        self.iobutton_frame.grid(row=0, column=0, sticky='nsew')
        self.button_frames.append(self.iobutton_frame)
        self.all_frames.append(self.iobutton_frame)

        # add widgets for plotting
        self.plotbutton_frame = tk.Frame(master)
        self.plotbutton_label = tk.Label(self.plotbutton_frame, text='Plotting')
        self.plotbutton_label.grid(row=0, column=0)
        self.plot_sets_button = tk.Button(self.plotbutton_frame, text='Plot',
                                          command=self.open_plot_sets_window)
        self.plot_sets_button.grid(row=1, column=0)
        # to do: add button and functionality for plotting runs/lumisections
        # add the frame to the window
        self.plotbutton_frame.grid(row=1, column=0, sticky='nsew')
        self.button_frames.append(self.plotbutton_frame)
        self.all_frames.append(self.plotbutton_frame)

        # add widgets for resampling
        self.resampling_frame = tk.Frame(master)
        self.resampling_label = tk.Label(self.resampling_frame, text='Resampling')
        self.resampling_label.grid(row=0, column=0)
        self.resample_button = tk.Button(self.resampling_frame, text='Resample',
                                    command=self.open_resample_window)
        self.resample_button.grid(row=1, column=0)
        # add the frame to the window
        self.resampling_frame.grid(row=2, column=0, sticky='nsew')
        self.button_frames.append(self.resampling_frame)
        self.all_frames.append(self.resampling_frame)

        # add widgets for classifier training, fitting and evaluation
        self.model_frame = tk.Frame(master)
        self.model_label = tk.Label(self.model_frame, text='Model building')
        self.model_label.grid(row=0, column=0)
        self.train_button = tk.Button(self.model_frame, text='Train classifiers',
                                      command=self.open_train_window)
        self.train_button.grid(row=1, column=0)
        self.apply_classifiers_button = tk.Button(self.model_frame, 
                                        text='Evaluate classifiers',
                                        command=self.open_apply_classifiers_window)
        self.apply_classifiers_button.grid(row=2, column=0)
        self.fit_button = tk.Button(self.model_frame, text='Fit', 
                                    command=self.open_fit_window)
        self.fit_button.grid(row=3, column=0)
        self.apply_fit_button = tk.Button(self.model_frame,
                                            text='Evaluate fit',
                                            command=self.open_apply_fit_window)
        self.apply_fit_button.grid(row=4, column=0)
        self.evaluate_button = tk.Button(self.model_frame, text='Evaluate model',
                                                command=self.open_evaluate_window)
        self.evaluate_button.grid(row=5, column=0)
        # add the frame to the window
        self.model_frame.grid(row=3, column=0, sticky='nsew')
        self.button_frames.append(self.model_frame)
        self.all_frames.append(self.model_frame)

        # add widgets for displaying text
        self.stdout_frame = tk.Frame(master)
        self.stdout_frame.grid(row=0, column=1, rowspan=len(self.button_frames),
                                sticky='nsew')
        self.all_frames.append(self.stdout_frame)
        self.stdout_label = tk.Label(self.stdout_frame, text='Stdout')
        self.stdout_label.grid(row=0, column=0)
        self.messages_text = ScrolledTextFrame(self.stdout_frame, txtwidth=50, txtheight=25)
        initstring = 'Welcome to the ML4DQM GUI!\n'
        self.messages_text.widget.insert(tk.INSERT, initstring)
        self.messages_text.frame.grid(row=1, column=0)

        # redirect stdout (and stderr) to text widget
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = StdOutRedirector( self.messages_text.widget, self.master )
        #sys.stderr = StdOutRedirector( self.messages_text.widget, self.master )

        # add widgets for displaying HistStruct info
        self.histstruct_info_frame = tk.Frame(master)
        self.histstruct_info_frame.grid(row=0, column=2, rowspan=len(self.button_frames),
                                        sticky='nsew')
        self.all_frames.append(self.histstruct_info_frame)
        self.histstruct_info_label = tk.Label(self.histstruct_info_frame, 
                                            text='HistStruct info')
        self.histstruct_info_label.grid(row=0, column=0, columnspan=2)
        self.histstruct_filename_label = tk.Label(self.histstruct_info_frame, text='File')
        self.histstruct_filename_label.grid(row=1, column=0, columnspan=2)
        self.histstruct_filename_text = ScrolledTextFrame(self.histstruct_info_frame,
                                            txtwidth=45, txtheight=1)
        self.histstruct_filename_text.frame.grid(row=2, column=0, columnspan=2)
        initstring = '[no histstruct loaded]'
        self.histstruct_filename_text.widget.insert(tk.INSERT, initstring)
        self.histstruct_masknames_label = tk.Label(self.histstruct_info_frame, text='Masks')
        self.histstruct_masknames_label.grid(row=3, column=0)
        self.histstruct_masknames_text = ScrolledTextFrame(self.histstruct_info_frame, 
                                            txtwidth=25, txtheight=15)
        self.histstruct_masknames_text.frame.grid(row=4, column=0)
        initstring = '[no histstruct loaded]'
        self.histstruct_masknames_text.widget.insert(tk.INSERT, initstring)
        self.histstruct_extnames_label = tk.Label(self.histstruct_info_frame, 
                                            text='Extended sets')
        self.histstruct_extnames_label.grid(row=3, column=1)
        self.histstruct_extnames_text = ScrolledTextFrame(self.histstruct_info_frame, 
                                            txtwidth=25, txtheight=15)
        self.histstruct_extnames_text.frame.grid(row=4, column=1)
        initstring = '[no histstruct loaded]'
        self.histstruct_extnames_text.widget.insert(tk.INSERT, initstring)

        # apply default stylings to all frames
        for frame in self.all_frames:
            set_frame_default_style( frame )

        # apply default stylings to button frames
        for frame in self.button_frames:
            pass

    def clear_histstruct_info(self):
        ### clear all widgets displaying histstruct info
        self.histstruct_filename_text.widget.delete(1.0, tk.END)
        self.histstruct_masknames_text.widget.delete(1.0, tk.END)
        self.histstruct_extnames_text.widget.delete(1.0, tk.END)

    def update_histstruct_info(self):
        self.clear_histstruct_info()
        self.histstruct_filename_text.widget.insert(tk.INSERT, self.histstruct_filename)
        masknames = self.histstruct.get_masknames()
        if len(masknames)>0: 
            self.histstruct_masknames_text.widget.insert(tk.INSERT, '\n'.join(masknames))
        else: self.histstruct_masknames_text.widget.insert(tk.INSERT, '[no masks available]')
        extnames = self.histstruct.exthistograms.keys()
        if len(extnames)>0: 
            self.histstruct_extnames_text.widget.insert(tk.INSERT, '\n'.join(extnames))
        else: self.histstruct_extnames_text.widget.insert(tk.INSERT, '[no sets available]')

    def load_histstruct(self):
        initialdir = os.path.abspath(os.path.dirname(__file__))
        filename = fldlg.askopenfilename(initialdir=initialdir,
                    title='Load a HistStruct',
                    filetypes=(('zip files','*.zip'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0: 
            print('Loading of HistStruct canceled')
            return
        # clear current histstruct and related info before loading new one
        self.histstruct = None
        self.clear_histstruct_info()
        # load a new histstruct
        self.histstruct = HistStruct.HistStruct.load( filename, verbose=True )
        self.histstruct_filename = filename
        # fill related widgets
        self.update_histstruct_info()

    def save_histstruct(self):
        initialdir = os.path.abspath(os.path.dirname(__file__))
        filename = fldlg.asksaveasfilename(initialdir=initialdir,
                    title='Save a HistStruct',
                    filetypes=(('zip files','*.zip'),('all files','*.*')))
        # if filename is invalid, return
        if len(filename)==0:
            print('Saving of HistStruct canceled')
            return
        # save the histstruct
        self.histstruct.save( filename )

    def open_plot_sets_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = PlotSetsWindow(self.master, self.histstruct)
        return

    def open_train_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = TrainClassifiersWindow(self.master, self.histstruct)
        return

    def open_fit_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = FitWindow(self.master, self.histstruct)
        return

    def open_resample_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = ResampleWindow(self.master, self.histstruct)
        return

    def open_evaluate_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = EvaluateWindow(self.master, self.histstruct)
        return

    def open_display_histstruct_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = DisplayHistStructWindow(self.master, self.histstruct)
        return

    def open_apply_classifiers_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = ApplyClassifiersWindow(self.master, self.histstruct)
        return

    def open_apply_fit_window(self):
        if not self.histstruct:
            print('ERROR: need to load a HistStruct first')
            return
        _ = ApplyFitWindow(self.master, self.histstruct)
        return


if __name__=='__main__':

    window = tk.Tk()
    gui = ML4DQMGUI(window)
    window.mainloop()