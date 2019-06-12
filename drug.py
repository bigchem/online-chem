'''
   Drug Search program: recurrent neural networks
   and molecular docking.

   Helmholtz-Zentrum Munchen, STB
   P. Karpov, I. Tetko, 2018

   carpovpv@gmail.com
'''

import numpy as np
import sys as sys
import math as math

import chemfilter as cf
import rnnmodel as rn
import moldb

from rdkit import Chem

def main():

   if len(sys.argv) != 2:
       print("Usage: ", sys.argv[0], " file-with-original-smiles.");
       return;

   print("Starting the program.");

   Pretrained = "data/rnn.npy";
   ActiveMols = sys.argv[1];

   print("Loading active molecules: ", ActiveMols);

   chars = "$#%()+-./0123456789=@ABCFGHIKLMNOPRSTVXZ[\]abcdegilnoprstu^";
   chars = sorted(set(chars))

   print("Loading and building RNN functions.");
   rnn = rn.RnnModel(chars, Pretrained);

   print("Loading filters.");
   filters = [];

   #here we used only a Lipinski-like filter (MW < 700)
   filters.append( cf.LipinskiFilter() );

   #docking filter uses parallel docking on differen machines,
   #so it loads separately from other filters.
   bm = cf.VinaFilter();

   print("Loading and preparing the database.");
   db = moldb.MolDb();

   #load our actives
   #dont check uniqness. This allows to increase the training dataset.
   cnt = 0;
   with open(ActiveMols) as f:
      lines = f.readlines();
      for line in lines:
         m, canon = cf.checkSmile(line);
         if(m is not None):
            db.addMol(canon, line);
            cnt += 1;

   print("Loaded {} molecules.".format(cnt));


   temp_mols = [];
   #the generation clear our temporary table with smiles
   def rnn_before():
      temp_mols.clear();

   #after we calculated all scoring values
   #we simple add favourable molecules to the database
   def rnn_after(cycle):
      print("After docking:");
      print(temp_mols);
      cnt = 0;
      for k in temp_mols:
         if(k[0] == True and len(k) == 5):
            idx = db.addMol(k[1], k[3], True, cycle, k[2]);
            db.updateTargetValue(idx, k[4]);
      return;

   #generate a set of smiles from scratch
   #or starting with a particular fragment
   def rnn_generate(x):
      bait = "^";
      bait += x;
      x = bait;

      score = 0.0;
      raw = rnn.generate(x);

      new_mols = [];
      for line in raw:
         m, canon = cf.checkSmile(line);
         if(m is not None):

            was = False;
            for tm in temp_mols:
               if(canon == tm[1]):
                  temp_mols.append([False, canon, 0.0, line]);
                  was = True;
                  break;

            if was == True:
                continue;

            idx, oldscore, oldhits = db.search(canon);
            #1 found in actives, 2 found in inactives, 0 not found at all
            if(idx == 1 or idx == 2):
               continue;

            allow = True;
            try:
               for fl in filters:
                  if(fl.calcScore(m) == False):
                     allow = False;
                     break;
            except:
               print("Exception. Continue.", sys.exc_info()[0]);

            if allow == True:
                new_mols.append ( [m, canon, line] );

      if len(new_mols):

          bm.calcScore(new_mols);
          print("After docking:");

          #docking filter adds the VINA score to the end of the list on 3 position
          for idx in range(len(new_mols)):
             if(len(new_mols[idx]) > 3):
                temp_mols.append([True, new_mols[idx][1], new_mols[idx][3], new_mols[idx][2], new_mols[idx][3]]);
   #end rnn_generate_function

   #loop: training -> generation -> filtering and docking -> new data
   max_cycle = 10;

   #max_attempts must be sufficiently large to sample enough molecules with
   #high VINA scores. This new set will be used for further training of the Generator.
   max_attempts = 10;

   for cycle in range(1, max_cycle + 1):

      print("=======CYCLE {} =======".format(cycle));

      pretrain_without_improvement = 0;
      max_trials = 3;

      for pretrain in range(1, max_trials):

         data = db.selectMols();
         vals = rnn.fineTune(data);

         id_min = np.argmin(vals);
         if(id_min == 0):
            pretrain_without_improvement += 1;
            print("Nothing improved after training. Step: {}".format(pretrain_without_improvement));
         else:
            break;

      sys.stdout.flush();

      if(pretrain_without_improvement >= max_trials):
         print("Stop training. No improvement.");

      #save validation erros for the future analysis in epochs table.
      db.updateLearning(cycle, vals);
      
      #save weights each cycle
      rnn.save("weights/weights-{}".format(cycle));

      #increase the temperature each cycle
      rnn.setTemperature(1.0 + 0.05*(cycle  -1));

      print("Start generation for cycle ", cycle, ".");

      for i in range(max_attempts):
         rnn_before();
         #put here a fragment if you want to generate from a particular core
         rnn_generate("");
         rnn_after(cycle);
         db.commit();
         sys.stdout.flush();
      #Finish feneration within the cycle.

      nt = db.averageTarget(cycle);
      print("Average value for cycle {} is {}".format(cycle, nt));
      sys.stdout.flush();

   rnn.save("weights/weights-end");
   print("Finish procedure. Relax!");

if __name__ == "__main__" :
   main();
