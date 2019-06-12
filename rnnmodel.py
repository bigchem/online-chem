import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys

N_HIDDEN = 512
LEARNING_RATE = .01
GRAD_CLIP = 100
NUM_EPOCHS = 50

class RnnModel:

   def __init__(self, vocabulary, file_weights):

      self.vocab = vocabulary;
      self.vocab_size = len(self.vocab);

      self.char_to_ix = { ch:i for i,ch in enumerate(self.vocab) }
      self.ix_to_char = { i:ch for i,ch in enumerate(self.vocab) }

      self.start_seq = self.char_to_ix["^"];
      self.end_seq = self.char_to_ix["$"];

      self.Temperature = 0;

      self.pred, self.train, self.valid, self.params = \
           self.buildRecurrentNetwork(file_weights);

   def generate(self, bait):

      max_length = 300;
      batch_size = 8;

      yt = np.zeros((batch_size, max_length, self.vocab_size), np.int8);
      mt = np.zeros((batch_size, max_length), np.int8);

      for i in range(len(bait)):
         yt[:, i, self.char_to_ix[ bait[i] ] ] =1;
         mt[:, i] = 1;

      for y in range(len(bait), max_length):
         n = self.pred(yt, mt);
         p = n[: , y, :];
         mt[:, y] = 1;

         for batch in range(batch_size):
            prob = p[batch];
            if(self.Temperature > 0):
               p[batch] = np.log(p[batch]) / self.Temperature;
               prob = np.exp(p[batch]) / np.sum(np.exp(p[batch]));
            yt[batch, y, np.random.choice(np.arange(self.vocab_size), p= prob) ] = 1.0;

      res = [];
      for batch in range(batch_size):
         st = "";
         for q in range(max_length):
             w = np.argmax(yt[batch, q]);
             if(w == self.char_to_ix["$"]):
                 res.append(st);
                 break;
             if(q > 0):
                 st += self.ix_to_char[ w ];

      return res;

   #for training
   def gen_data(self, data):

       batch_size = len(data);
       seq_length = len(data[0]);

       for line in data:
           if len(line) > seq_length:
               seq_length = len(line);
       seq_length += 1;

       x = np.zeros((batch_size, seq_length, self.vocab_size), np.int8);
       y = np.zeros((batch_size, seq_length, self.vocab_size), np.int8);
       m = np.zeros((batch_size, seq_length), np.int8);

       cnt = 0;
       for line in data:

           line = line.replace("\n","");
           line = "^" + line + "$";

           #print(line, len(line));

           n = len(line) - 1;
           m[ cnt, : n ] = 1;

           for i in range(n):
              x[cnt, i, self.char_to_ix[ line[i]] ] = 1;
              y[cnt, i, self.char_to_ix[ line[i+1]] ] = 1;

           cnt += 1;

       return x, y, m;

   def setTemperature(self, Temperature):
      self.Temperature= Temperature;

   def buildRecurrentNetwork(self, weights):

      X = T.tensor3('X', dtype='int8');
      M = T.matrix('M', dtype='int8');
      Y = T.tensor3('Y', dtype='int8');

      l_in = lasagne.layers.InputLayer(shape=(None, None, self.vocab_size), input_var=X);
      l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=M);
      batchsize, seqlength,_ = l_in.input_var.shape;

      l_forward_1 = lasagne.layers.LSTMLayer(
          l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
          mask_input = l_mask,
          nonlinearity=lasagne.nonlinearities.tanh,
          only_return_final = False)

      l_forward_2 = lasagne.layers.LSTMLayer(
          l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
          nonlinearity=lasagne.nonlinearities.tanh,
          mask_input = l_mask,
          only_return_final = False)

      l_shp = lasagne.layers.ReshapeLayer(l_forward_2, (-1, N_HIDDEN));
      l_soft = lasagne.layers.DenseLayer(l_shp, num_units=self.vocab_size, W = lasagne.init.Normal(),
                                         nonlinearity=lasagne.nonlinearities.softmax)
      l_out = lasagne.layers.ReshapeLayer(l_soft, (batchsize, seqlength, self.vocab_size));

      network_output = lasagne.layers.get_output(l_out, deterministic = False)
      network = lasagne.layers.get_output(l_out, deterministic = True)

      cost = T.nnet.categorical_crossentropy(network_output, Y);
      cost = cost * M;
      cost = cost.mean();

      all_params = lasagne.layers.get_all_params(l_out,trainable=True)

      print("Computing updates ...")
      updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

      print("Compiling functions ...")
      train = theano.function([X, M, Y], cost, updates=updates, allow_input_downcast=True)
      compute_cost = theano.function([X, M, Y], cost, allow_input_downcast=True)

      probs = theano.function([X, M],network,allow_input_downcast=True)

      params = np.load(weights);
      lasagne.layers.set_all_param_values(l_out, params);

      return probs, train, compute_cost, l_out;


   def fineTune(self, data):

      val_lost = [];

      #11% of the whole data for validation
      TV = 0.11;
      num_epochs = 5;

      total = len(data);
      tv = int(TV * total);

      training = data[:total - tv];
      validation = data [-tv:];

      init_val = lasagne.layers.get_all_param_values(self.params);

      bestval = None;
      params = None;

      improved = False;

      for it in range(1, num_epochs + 1):
         avg_cost = 0.0;
         avg_val = 0.0;

         lines = [];
         for o in range(len(training)):

            if len(lines) > 16:
               x,y,m = self.gen_data(lines);
               avg_cost += self.train(x, m, y);
               lines = [];
            lines.append(training[o]);

         if len(lines) > 0:
            x,y,m = self.gen_data(lines);
            avg_cost += self.train(x, m, y);
            lines = [];

         lines = [];
         for o in range(len(validation)):
            if len(lines) > 16:
               x,y,m = self.gen_data(lines);
               avg_val += self.valid(x, m, y);
               lines = [];
            lines.append(validation[o]);

         if len(lines) > 0:
            x,y,m = self.gen_data(lines);
            avg_val += self.valid(x, m, y);

         val_lost.append(avg_val);

         if(it == 1):
            bestval = avg_val;
            params = lasagne.layers.get_all_param_values(self.params);
         else:
            if(avg_val < bestval):
               bestval = avg_val;
               params = lasagne.layers.get_all_param_values(self.params);
               improved = True;

         print("   Epoch {} average training loss = {}, validation loss = {}".format(it, avg_cost, avg_val));
         print(self.generate("^"));

      if(improved == True):
         lasagne.layers.set_all_param_values(self.params, params);
         print("Replacing the parameters of the network.");
      else:
         lasagne.layers.set_all_param_values(self.params, init_val);

      return val_lost;

   def save(self, name):

      params = lasagne.layers.get_all_param_values(self.params);
      np.save(name, params);
