# Online-Chem
Generation of new putative Mdmx inhibitors from scratch based on Recurrent Neural Networks and molecular docking.

# Dependencies

* Python3 https://www.python.org
* RdKit https://www.rdkit.org
* OpenBabel for converting to PDBQT file format http://openbabel.org/wiki/Main_Page
* Theano for numerical computation https://theano.readthedocs.io/en/0.8.x
* Lasagne library with deep-learning primitives for Theano https://lasagne.readthedocs.io/en/latest
* AutoDock Vina for molecular docking http://vina.scripps.edu
* Lilly Medchem rules for reactivity filtering https://github.com/IanAWatson/Lilly-Medchem-Rules

# Description of files

* data
  * mdmx.conf - a configuration file for AutoDock Vina with parameters of Mdmx binding site
  * protein.pdbqt - a pdbqt file with the receptor
  * rnn.npy - weights for the RNN-based Generator trained on ChEMBL database. It generates approximately 90% of valid SMILES
* weights - this directory contains weights of the RNN after each fine-tuning cycle.
* work - a working directory for molecular docking and any other temporary files.
* chemfilter.py - a place to put all additional filters including molecular docking. Currently only Lipinski-like filter is added. 
* dock-local.sh - a task for one-molecular docking. If you can a number of servers to do parallel docking than it is possible to clone this file and change the instructions. In this case, the workflow will consist of transfer the sdf file to another server, perform docking, and return the result. Vina filter uses Future objects to start docking simultaneously and receive the results later.
* drug.py - the main program.
* mdmx.smi - an augmented set of SMILES of Mdmx inhibitors collected from ChEMBL and BindingDB databases.
* moldb.py - SQLite functionality. All generated molecules and their corresponding results are stored in SQLite database. The result of the whole simulation is work.screening.db. It is possible to access the data with https://sqlitebrowser.org
* rnnmodel.py - the source code for the RNN generator model and function for further tuning it with newly generated data.
* vina.sh - a local script to perform actual molecular docking. 

# Algorithm

This code of the project is particulaly tuned to Mdmx target, so the adaptation to other targets may require some programming. 
The drug.py generates new molecules in cycles, each of them consists of the following steps:

1. Select available inhibitors from the work/screening.db keeping the ratio of original and generated compounds around 50%. 
1. Fine-tune the RNN model with early-stopping. 
1. Generate new structures and calculate all filters. For those molecules, which successfully pass the filters, the molecular docking estimates their final scores.
1. Select new putative inhibitors based on Vina scores and add them to the future training set.
1. Repeat

# Usage 

python3 drug.py mdmx.smi. 

Once the progam finished, the results are in work/screening.db. The table mols contains all the SMILES from the study as well as cycle of generation and Vina scores placed in tv and score columns.

