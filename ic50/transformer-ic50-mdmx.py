import sys
import os
import time
import pickle
import math
import numpy as np
import csv
import h5py
import tarfile
import shutil
import math
from http.server import HTTPServer, BaseHTTPRequestHandler

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from rdkit import Chem
from layers import PositionLayer, MaskLayerLeft, \
                   MaskLayerRight, MaskLayerTriangular, \
                   SelfLayer, LayerNormalization

N_HIDDEN = 512;
N_HIDDEN_CNN = 512;
EMBEDDING_SIZE = 64;
KEY_SIZE = EMBEDDING_SIZE;
SEQ_LENGTH = 512;

#our vocabulary
chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$";
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False);
config.gpu_options.allow_growth = True;
tf.logging.set_verbosity(tf.logging.ERROR);
K.set_session(tf.Session(config=config));

class suppress_stderr(object):
   def __init__(self):
       self.null_fds = [os.open(os.devnull,os.O_RDWR)]
       self.save_fds = [os.dup(2)]
   def __enter__(self):
       os.dup2(self.null_fds[0],2)
   def __exit__(self, *_):
       os.dup2(self.save_fds[0],2)
       for fd in self.null_fds + self.save_fds:
          os.close(fd)

def findBoundaries(DS):

    x = [];
    for i in range(len(DS)):
      x.append(DS[i][1]);

    hist= np.histogram(x)[0];
    if np.count_nonzero(hist) > 2:
       y_min = np.min(x);
       y_max = np.max(x);

       add = 0.01 * (y_max - y_min);
       y_max = y_max + add;
       y_min = y_min - add;

       print("regression:", y_min, "to", y_max, "scaling...");

       for i in range(len(DS)):
          DS[i][1] = 0.9 + 0.8 * (DS[i][1] - y_max) / (y_max - y_min);

       return ["regression", y_min, y_max];

    else:
       print("classification");
       return ["classification"];

def analyzeDescrFile(fname):

    first_row = True;

    DS = [];
    ind_mol = 0;
    ind_val = 1;

    for row in csv.reader(open(fname, "r")):

       mol = row[ind_mol];
       #print(row);
       val = float(row[ind_val]);

       arr = [];
       if CANONIZE == 'True':
          with suppress_stderr():
             m = Chem.MolFromSmiles(mol);
             if m is not None:
                for step in range(100):
                   arr.append(Chem.MolToSmiles(m, rootedAtAtom = np.random.randint(0, m.GetNumAtoms()), canonical = False));
             else:
                 arr.append(mol);
       else:
          arr.append(mol);

       arr = list(set(arr));
       for step in range(len(arr)):
          DS.append( [ arr[step], float(val) ]);

    info = findBoundaries(DS);
    return [DS, info];

def gen_data(data, nettype="regression"):

    batch_size = len(data);

    #search for max lengths
    nl = len(data[0][0]);
    for i in range(1, batch_size, 1):
        nl_a = len(data[i][0]);
        if nl_a > nl:
            nl = nl_a;

    if nl >= SEQ_LENGTH:
        raise Exception("Input string is too long.");

    x = np.zeros((batch_size, SEQ_LENGTH), np.int8);
    mx = np.zeros((batch_size, SEQ_LENGTH), np.int8);
    z = np.zeros((batch_size) if (nettype == "regression") else (batch_size, 2), np.float32);

    for cnt in range(batch_size):

        n = len(data[cnt][0]);
        for i in range(n):
           x[cnt, i] = char_to_ix[ data[cnt][0][i]] ;
        mx[cnt, :i+1] = 1;

        if nettype == "regression":
           z [cnt ] = data[cnt][1];
        else:
           z [cnt , int(data[cnt][1]) ] = 1;

    return [x, mx], z;


def data_generator(ds, nettype = "regression"):

   data = [];
   while True:
      for i in range(len(ds)):
         data.append( ds[i] );
         if len(data) == BATCH_SIZE:
            yield gen_data(data, nettype);
            data = [];
      if len(data) > 0:
         yield gen_data(data, nettype);
         data = [];

def buildNetwork(unfreeze, nettype):

    n_block, n_self = 3, 10;

    l_in = layers.Input( shape= (SEQ_LENGTH,));
    l_mask = layers.Input( shape= (SEQ_LENGTH,));

    #transformer part
    #positional encodings for product and reagents, respectively
    l_pos = PositionLayer(EMBEDDING_SIZE)(l_mask);
    l_left_mask = MaskLayerLeft()(l_mask);

    #encoder
    l_voc = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None, trainable = unfreeze);
    l_embed = layers.Add()([ l_voc(l_in), l_pos]);

    for layer in range(n_block):

       #self attention
       l_o = [ SelfLayer(EMBEDDING_SIZE, KEY_SIZE, trainable= unfreeze) ([l_embed, l_embed, l_embed, l_left_mask]) for i in range(n_self)];

       l_con = layers.Concatenate()(l_o);
       l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE, trainable = unfreeze), trainable = unfreeze) (l_con);
       if unfreeze == True: l_dense = layers.Dropout(rate=0.1)(l_dense);
       l_add = layers.Add()( [l_dense, l_embed]);
       l_att = LayerNormalization(trainable = unfreeze)(l_add);

       #position-wise
       l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu', trainable = unfreeze)(l_att);
       l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1, trainable = unfreeze)(l_c1);
       if unfreeze == True: l_c2 = layers.Dropout(rate=0.1)(l_c2);
       l_ff = layers.Add()([l_att, l_c2]);
       l_embed = LayerNormalization(trainable = unfreeze)(l_ff);

    #end of Transformer's part
    l_encoder = l_embed;

    #text-cnn part
    #https://github.com/deepchem/deepchem/blob/b7a6d3d759145d238eb8abaf76183e9dbd7b683c/deepchem/models/tensorgraph/models/text_cnn.py

    kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
    num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160];

    l_pool = [];
    for i in range(len(kernel_sizes)):
       l_conv = layers.Conv1D(num_filters[i], kernel_size=kernel_sizes[i], padding='valid', kernel_initializer='normal', activation='relu')(l_encoder);
       l_maxpool = layers.Lambda(lambda x: tf.reduce_max(x, axis=1))(l_conv);
       l_pool.append(l_maxpool);

    l_cnn = layers.Concatenate(axis=1)(l_pool);
    l_cnn_drop = layers.Dropout(rate = 0.25)(l_cnn);

    #dense part
    l_dense =layers.Dense(N_HIDDEN_CNN, activation='relu') (l_cnn_drop);

    #High Way unit
    transform_gate = layers.Dense(units= N_HIDDEN_CNN, activation="sigmoid",
                     bias_initializer=tf.keras.initializers.Constant(-1))(l_dense);

    carry_gate = layers.Lambda(lambda x: 1.0 - x, output_shape=(N_HIDDEN_CNN,))(transform_gate);
    transformed_data = layers.Dense(units= N_HIDDEN_CNN, activation="relu")(l_dense);
    transformed_gated = layers.Multiply()([transform_gate, transformed_data]);
    identity_gated = layers.Multiply()([carry_gate, l_dense]);
    l_highway = layers.Add()([transformed_gated, identity_gated]);

    if nettype == "regression":
       l_out = layers.Dense(1, activation='linear', name="Regression") (l_highway);
       mdl = tf.keras.Model([l_in, l_mask], l_out);
       mdl.compile (optimizer = 'adam', loss = 'mse', metrics=['mse'] );
    else:
       l_out = layers.Dense(2, activation='softmax', name="Classification") (l_highway);
       mdl = tf.keras.Model([l_in, l_mask], l_out);
       mdl.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'] );

    #auxiliary model fomr the Encoder part of Transformer
    if unfreeze == False:
       mdl2 = tf.keras.Model([l_in, l_mask], l_encoder);
       mdl2.set_weights(np.load("embeddings.npy"));

    K.set_value(mdl.optimizer.lr, 1e-4);

    #plot_model(mdl, to_file='model.png', show_shapes=True);
    #mdl.summary();

    return mdl;

if __name__ == "__main__":

   tar = tarfile.open('../weights/ic50-1.tar');
   tar.extractall();
   tar.close();

   info = open("model.txt").read().strip();
   info = info.replace("'", "");
   info = info.replace("[", "");
   info = info.replace("]", "").split(",");

   nettype = info[0].strip();
   mdl1 = buildNetwork(True, nettype);
   mdl1.load_weights("model.h5");

   tar = tarfile.open('../weights/ic50-2.tar');
   tar.extractall();
   tar.close();

   mdl2 = buildNetwork(True, nettype);
   mdl2.load_weights("model.h5");

   os.remove("model.txt");
   os.remove("model.h5");

   if nettype == "regression":
      info[1] = float(info[1]);
      info[2] = float(info[2]);
   else:
      info[1] = float(info[1]);

   print("Start the server");

   #start listener here
   class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
      def do_GET(self):
         self.send_response(200);
         self.end_headers();

         mol = self.path[1:];
         print(mol);

         arr = [];
         with suppress_stderr():
            m = Chem.MolFromSmiles(mol);
            if m is not None:
               for step in range(100):
                 arr.append(Chem.MolToSmiles(m, rootedAtAtom = np.random.randint(0, m.GetNumAtoms()), canonical = False));
            else:
               arr.append(mol);

         arr = list(set(arr));
         d = [];
         for step in range(len(arr)):
            d.append([arr[step], 0]);

         x, y = gen_data(d, nettype);

         y1 = np.mean(mdl1.predict(x));
         y2 = np.mean(mdl2.predict(x));

         y = (y1 + y2) /2.0;
         if nettype == "regression":
            y = (y - 0.9) / 0.8 * (info[2] - info[1]) + info[2];

         self.wfile.write(str(y).encode());
         #end get

   httpd = HTTPServer(('', 3000), SimpleHTTPRequestHandler)
   httpd.serve_forever();
