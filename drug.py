'''
   Drug Search program: recurrent neural networks,
   monte-carlo search trees and molecular docking.

   Helmholtz-Zentrum Munchen, STB
   P. Karpov, I. Tetko, 2018-2019

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
       print("Usage: ", sys.argv[0], " molecules.smi");
       return;

   print("Starting the program.");

   Pretrained = "data/rnn.npy";
   ActiveMols = sys.argv[1];

   print("Loading active molecules: ", ActiveMols);
   print("Loading the vocabulary.");

   chars = open('data/chars.txt', 'r').read()[:-1];
   chars = sorted(set(chars))

   print("Loading and building RNN functions.");
   rnn = rn.RnnModel(chars, Pretrained);

   print("Loading filters.");
   filters = [];

   ic50 = cf.IC50Filter();

   filters.append( cf.LipinskiFilter() );
   filters.append( cf.LillyFilter() );
   filters.append( cf.PainsFilter() );
   filters.append( ic50 );

   bm = cf.VinaFilter();

   print("Loading and preparing the database.");
   db = moldb.MolDb();

   cnt = 0;
   #load our actives
   #dont check uniqness. This allows to increase the training dataset.

   with open(ActiveMols) as f:
      lines = f.readlines();
      for line in lines:
         line = line.strip();
         m, canon = cf.checkSmile(line);
         if(m is not None):
            db.addMol(canon, line);
            cnt += 1;

   print("Loaded {} molecules.".format(cnt));

   temp_mols = [];
   def rnn_before():
      temp_mols.clear();

   def rnn_after(cycle):
      print(temp_mols);
      cnt = 0;
      for k in temp_mols:
         if(k[0] == True and len(k) == 6):
            idx = db.addMol(k[1], k[3], True, cycle, k[2]);
            db.updateTargetValue(idx, k[4], k[5]);
      return;

   #apply state in the MC workflow
   #the function returns the score
   def rnn_generate(x):
      bait = "^";
      bait += x;
      x = bait;

      score = 0.0;
      raw = rnn.generate(x);

      print("Generated: ", raw);

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

            if(idx == 1 or idx == 2):
               continue;

            allow = True;

            try:

               for fl in filters:
                  if(fl.calcScore(m, canon) == False):
                     allow = False;
                     break;
            except:
               print("Exception. Continue.", sys.exc_info()[0]);

            if allow == True:
                new_mols.append ( [m, canon, line, ic50.value] );

      if len(new_mols):

          bm.calcScore(new_mols);
          print("After docking:");

          for idx in range(len(new_mols)):
             if(len(new_mols[idx]) > 4):
                temp_mols.append([True, new_mols[idx][1], new_mols[idx][3], new_mols[idx][2], new_mols[idx][4], new_mols[idx][3]]);

   #end rnn_generate_function

   max_cycle = 10;
   mtree_max = 1000;

   ##!!!
   for cycle in range(1, max_cycle + 1):

      print("=======CYCLE {} =======".format(cycle));

      if cycle > 0:
         pretrain_without_improvement = 0;
         max_trials = 3;

         for pretrain in range(1, max_trials):

            data = db.selectMols();

            #every cycle decrease the learning rate
            lr = 0.01 * math.exp(-(cycle -1.0) / 4.0);
            rnn.setLearningRate(lr);

            #train the model on new data
            vals = rnn.fineTune(data);

            id_min = np.argmin(vals);
            if(id_min == 0):
               pretrain_without_improvement += 1;
               print("Nothing improved after training. Step: {}".format(pretrain_without_improvement));
            else:
               break;

         sys.stdout.flush();

         if(pretrain_without_improvement >= max_trials):
            print("No improvement during training.");
            #break;

         #update only last values
         db.updateLearning(cycle, vals);

         #save weights each cycle
         rnn.save("weights/weights-{}".format(cycle));

      #increase the temperature each cycle
      rnn.setTemperature(1.0 + 0.05 * (cycle - 1) );
      print("Start generation.");

      for i in range(mtree_max):
         rnn_before();
         rnn_generate("");
         rnn_after(cycle);

         db.commit();
         sys.stdout.flush();

      nt = db.averageTarget(cycle);
      print("Average value for cycle {} is {}".format(cycle, nt));
      sys.stdout.flush();

   rnn.save("weights/weights-end");
   print("Finish procedure. Relax!");

if __name__ == "__main__" :
   main();
