
from subprocess import Popen, PIPE

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem, rdReducedGraphs
from rdkit import DataStructs
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import os
import sys
import os.path
import math

import theano
import theano.tensor as T
from theano import shared
import lasagne

#simple helper function to run an external shell command
def run_external(cmd):

   proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
   out, err = proc.communicate()
   exitcode = proc.returncode

   return exitcode, out, err


def checkSmile(smile):
   canon = "";
   m = Chem.MolFromSmiles(smile);
   if(m is not None):
      canon = Chem.MolToSmiles(m);
   return m, canon;

#The main idea of a Filter is to allow the compound to pass further.
#It returns True or False. THe first case will launch other filters,
#in the second, the molecule will be discarded.

class LipinskiFilter:
   #MW differs from original Rule of Five.
   #original ligands are more heavy
   def __init__(self):
      pass;

   def calcScore(self, m):

      mw = Descriptors.MolWt(m);

      #we also are not interested in very small compounds
      if mw > 700 or mw < 100: return False;

      num_hdonors = Lipinski.NumHDonors(m);
      num_hacceptors = Lipinski.NumHAcceptors(m);

      if num_hdonors > 5: return False;
      if num_hacceptors > 10: return False;

      return True;

#Vina molecular docking filter.
class VinaFilter:

   def __init__(self):
      pass;

   def calcScore(self, mols ):

      #here it is possible to add other servers for docking
      docks = ["./dock-local.sh"];
      docs_res = [];

      curdock = 0;
      executor = ProcessPoolExecutor(len(docks));
      futures = [];

      #array of flags. 1 if the docking succeeded
      fm = [];

      for idx in range(len(mols)):
         m = mols[idx][0];

         try:
            if(AllChem.EmbedMolecule(m,useRandomCoords=True) == 0):
               AllChem.MMFFOptimizeMolecule(m)
               if(m):

                  writer = Chem.SDWriter('work/mol' +str(idx) + '.sdf')
                  writer.write(m);
                  writer.close();

                  print("Submit ", docks[curdock], "work/mol" + str(idx) + ".sdf");
                  fut = executor.submit(run_external, [docks[curdock], "work/mol" + str(idx) + ".sdf"]);

                  curdock += 1;
                  if curdock > len(docks)-1:
                     curdock = 0;

                  futures.append(fut);
                  fm.append(1);
               else:
                 futures.append("");
                 fm.append(0);
            else:
               futures.append("");
               fm.append(0);
         except:
            futures.append("");
            fm.append(0);

      for idx in range(len(mols)):

          if fm[idx] == 1:
             res = futures[idx].result();

             exitcode, out, err = res;
             if(exitcode == 0):
                try:
                   out = str(out);
                   out = out.replace("b'", "");
                   out = out.replace("\\n'", "");

                   vs = float(out);
                   mols[idx].append(vs);

                except:
                   mols[idx].append(0);
