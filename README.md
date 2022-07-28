# Tests and Examples for ML4DQM/DC

This fork contains some example code and experiments for the ML4DQM/DC project.

The original repository contains some example code that has been used to create the projects in this fork. The basic concept behind much of the work here is using autoencoders to detect anomalies in per-lumisection histograms for increased granularity and automation. Specifically, the projects here focus on finding ways to merge histograms (thus training fewer models) or on ways to determine what data is most important for classifying histograms. 

# Before you Begin: Make sure to run the `DataPreprocess.ipynb` notebook in `Support Notebooks` to ensure data is available for use

In more detail:

- The projects here are primarily designed to support 1D histograms, however 2D implementation should not be a difficult task. 
- With the exception of the example code in the tutorials folder, primarily these notebooks use a custom autoencoder wtih a concatenate layer (known as Concatamash). This method of combining histograms is designed to decrease resource consumption and make up for insufficient data should a situation like that occur. 

## Structure of this repository:

### New Directories in Fork:

There are 3 key directories in this fork that don't exist in the original repository: Merging Histogram Notebooks, KH-AutoencoderTest, and Support Notebooks.
In more detail:

- `Merging Histogram Notebooks`: contains the notebooks for each different histogram merging method (1D Mash and Concatamash) as well as some variations of the two. The projects contained here include experimental implementations of the different methods.
- `KH-AutoencoderTest`: contains experiments related to this project. See readme inside directory for more information on each experiment.
- `Support Notebooks`: contains data pre/post processing notebooks to analyze in detail what is going into and coming out of trained autoencoders. 

### Expansions in Fork:

This fork also contains expansions of the following directories: utils, src. See original repository for more information on original implementation.
In more detail:
- `utils`: contains some new methods in existing utilities to provide new functionality. Original functionality is preserved with the addition of this new content.
- `src`: contains some new classes to provide support for different experiments:
    - `SubHistStruct`: a child class of HistStruct which implements the capacity for handling merged histograms. Most error checking and basic structure is preserved.
    - `FlexiStruct`: a child class of HistStruct with all the abilities of SubHistStruct, but with less error checking. This is necessary for using larger datasets the HistStruct was originally intended for and actively corrects for data discrepencies rather than preventing them. This allows for more complex operations as the expense of less strict data validation. 
    
## Tutorials:  

Some tutorials are located in the tutorials folder in this repository, that should help you get started with the code. They can be grouped into different steps:  

- Step 1: put the data in a more manageable format. The raw csv files that are our common input are not very easy to work with. Therefore you would probably first want to do something similar to what's done in the notebook `read_and_write_data.ipynb`. See the code and inline comments in that script and the functions it refers to for more detailed explanation. Its output is one single csv file per histogram type and per year, which is often much more convenient than the original csv files (which contain all histogram types together and are split per number of lines, not per run). All other functions and notebooks presuppose this first step.  
- Step 2: plot the data. Next, you can run `plot_histograms.ipynb` and `plot_histograms_loop.ipynb`. These notebooks should help you get a feeling of what your histogram looks like in general, and perhaps help you find some anomalies that you can use for testing. For 2D histograms, look at `plot_histograms_2d.ipynb` instead.  
- Step 3: train an autoencoder. The scripts autoencoder.ipynb and `autoencoder_iterative.ipynb` are used to train an autoencoder on the whole dataset or a particular subset respectively. Next, `autoencoder_combine.ipynb` and `1 trains autoencoders on multiple types of histograms and combines the mse's for each. An example on how to implement another classification method is shown in `template_combine.ipynb`.
- Step 4: (specific to this fork) navigate to the `Merging Histograms Notebooks` directory. Run the different cells of the `Concatamash.ipynb` notebook to see how a concatenate layer provides different functionality and merging characteristics to the autoencoders. Pay special attention to the histnames variable, which is a list of lists. Each sublist contains the histograms which will be combined in a single autoencoder. Therefore, the number of sublists is the number of autoencoders to be created and trained. 


### To get the tutorial notebooks running in SWAN  
#### (preferred method):  

- Log in to SWAN.  
- Go to Projects.  
- Click the cloud icon that says 'Download Project from git'  
- Paste the following url: [https://github.com/LukaLambrecht/ML4DQM-DC.git](https://github.com/LukaLambrecht/ML4DQM-DC.git).

#### (alternative method):  

- Log in to SWAN.
- Click on the leftmost icon on the top right ('new terminal').
- Navigate to where you want this repository (the starting place is your CERNBox home directory).
- Paste this command: git clone https://github.com/LukaLambrecht/ML4DQM-DC.git (or however you usually clone a repository).    
- Exit the terminal.  
- The folder should now be where you cloned it, and you can open and run the notebooks in it in SWAN. 
 
### Further documentation:  

- Documentation for all the class definitions and functions in the relevant code directories can be found [here](https://LukaLambrecht.github.io/ML4DQM-DC/).
- Note that the website above does not include documentation for the tutorials (yet?). However, some comments in the tutorial notebooks should provide (enough?) explanation to follow along.  
