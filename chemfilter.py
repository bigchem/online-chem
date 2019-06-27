
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
import csv
import os.path
import math

import theano
import theano.tensor as T
from theano import shared
import lasagne
import requests

def run_external(cmd):

   print(cmd);

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

class BasicFilter:
   def __init__(self, name):
      self.name = name;
      self.value = 0.0;

   def calcScore(self, m, smi):
      return True;

class IC50Filter(BasicFilter):

   def __init__(self):
      BasicFilter.__init__(self, "IC50Filter");

   def calcScore(self, m, smi):
      self.value = 0.0;

      try:

         r = requests.get("http://127.0.0.1:3000/" + smi);
         if r.status_code == 200:
            out = str(r.content);
            out = out.replace("b'", "");
            out = out.replace("'", "");
            self.value = float(out);
         else:
             return False;

         print("IC50 value: ", self.value);

         if self.value < 5:
             return False;

      except:
         print("Exception. Continue.", sys.exc_info()[0]);
         return False;

      return True;

class LipinskiFilter(BasicFilter):

   def __init__(self):
      BasicFilter.__init__(self, "Lipinsky");

   def calcScore(self, m, smi):

      self.value = 0.0;

      mw = Descriptors.MolWt(m);

      if mw > 700 or mw < 100: return False;

      num_hdonors = Lipinski.NumHDonors(m);
      num_hacceptors = Lipinski.NumHAcceptors(m);

      if num_hdonors > 5: return False;
      if num_hacceptors > 10: return False;


      return True;

class PainsFilter(BasicFilter):

   def __init__(self):
      self.painsFile = 'data/wehi_pains.csv';
      with open(self.painsFile, 'r') as inf:
         self.painsDefs = [x for x in csv.reader(inf)]
         self.rules = [Chem.MolFromSmarts(x[0], mergeHs=True) for x in self.painsDefs]

      print("Loaded ", len(self.rules), " smart Pains patterns.");
      BasicFilter.__init__(self, "PainsFilter");

   def calcScore(self, m, smi):
       self.value = 0.0;
       if '.' in smi: return False;
       for rule in self.rules:
         if m.HasSubstructMatch(rule) == True:
            return False;
       return True;

class LillyFilter(BasicFilter):
   def __init__(self):
      BasicFilter.__init__(self, "LillyFilter");

   def calcScore(self, m, smi):

      writer = Chem.SmilesWriter('work/mol-lilly.smi')
      writer.write(m);
      writer.close();

      exit_code, out, err = run_external(["./lilly.sh", os.getcwd() + "/work/mol-lilly.smi"]);
      os.unlink("work/mol-lilly.smi");
      
      if len(out) == 0:
          print("Not passed Lilly filter.");
          return False;

      return True;

class VinaFilter:
   def __init__(self):
      pass;
   def calcScore(self, mols):

      docks = ["./dock-local.sh" ];
      docs_res = [];

      curdock = 0;
      executor = ProcessPoolExecutor(len(docks));
      futures = [];
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
