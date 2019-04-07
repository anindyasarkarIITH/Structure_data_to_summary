

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import random
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch_field = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_field') # encoded input token for field
    self._enc_batch_content = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_content') # encoded input token for content
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')
    
    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    if hps.mode=="decode":
      #self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')
      self.prev_hybrid_attns = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_hybrid_attns')
      self.output_state = tf.placeholder(tf.float32, [hps.batch_size, hps.hidden_dim], name='output_state')
      #self.prev_hybrid_attns = list()
      #self.output_state = list()

  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch_field] = batch.enc_batch_field
    feed_dict[self._enc_batch_content] = batch.enc_batch_content
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, hidden_dim].
      encoder_final_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      #cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, encoder_final_states) = tf.nn.dynamic_rnn(cell, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      #encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, encoder_final_states


  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _add_decoder(self, inputs,emb_enc_inputs,emb_enc_inputs_fields,vsize_field,enc_batch_field_link):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings) is the initial encoder input token,e.g.,<start> token. A tensor of shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
    dec_lstm_hidden_dim = hps.hidden_dim
    #prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    #prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None
    prev_hybrid_attn = self.prev_hybrid_attns if hps.mode=="decode" else None
    cell_output = self.output_state if hps.mode=="decode" else None
    outputs,copy_scores,prev_hybrid_attns,output_state,decoder_state,p_gens = attention_decoder(inputs,emb_enc_inputs,emb_enc_inputs_fields, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell,dec_lstm_hidden_dim,vsize_field,enc_batch_field_link,cell_output, initial_state_attention=(hps.mode=="decode"),prev_hybrid_attn = prev_hybrid_attn,pointer_gen=hps.pointer_gen)

    #outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs,emb_enc_inputs,emb_enc_inputs_fields, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(hps.mode=="decode"))
    
    #return outputs, out_state, attn_dists, p_gens, coverage
    return outputs,copy_scores,prev_hybrid_attns,output_state,decoder_state ,p_gens
    
  def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]
      
      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      print (self._enc_batch_extend_vocab)
      shape = [self._hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize_field = self._vocab.size_field() # size of the vocabulary for field
    #vsize_field = self._vocab._count
    print ("@@@@@")
    print (vsize_field)
    vsize_content_summary = self._vocab.size()
    enc_batch_field_link = self._enc_batch_field

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
      
      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding_field = tf.get_variable('embedding_field', [vsize_field, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init) # field_embedding
        embedding_content_summary = tf.get_variable('embedding_content_summary', [vsize_content_summary, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init) # content embedding
        if hps.mode=="train": self._add_emb_vis(embedding_content_summary) # add to tensorboard
        emb_enc_inputs_fields = tf.nn.embedding_lookup(embedding_field, self._enc_batch_field) # tensor with shape (batch_size, max_enc_steps, emb_size) for fields
        emb_enc_inputs_contents = tf.nn.embedding_lookup(embedding_content_summary, self._enc_batch_content) # tensor with shape (batch_size, max_enc_steps, emb_size) for contents
        # concatenate the field embeddings and content embeddings
        emb_enc_inputs = tf.concat([emb_enc_inputs_fields, emb_enc_inputs_contents], axis=-1, name="concat")

        emb_dec_inputs = [tf.nn.embedding_lookup(embedding_content_summary, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
      
      # Add the encoder.
      enc_outputs, enc_final_states = self._add_encoder(emb_enc_inputs, self._enc_lens)
      print (enc_final_states) # shape lstmstateTuple of shape ((batch_size,hidden_dim_lstm),(batch_size,hidden_dim_lstm))
      self._enc_states = enc_outputs # shape (batch_size,max_encoder_step_for_this_batch,hidden_dim_lstm)
      
      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      #self._dec_in_state = self._reduce_states(fw_st, bw_st)
      self._dec_in_state = enc_final_states
      # Add the decoder.
      with tf.variable_scope('decoder'):
        decoder_outputs,copy_scores,self.prev_hybrid_attn,self.output_state_h,self._dec_out_state,self.p_gens = self._add_decoder(emb_dec_inputs,emb_enc_inputs,emb_enc_inputs_fields,vsize_field,enc_batch_field_link)
        # self.output_state -- h_prime_t, self.decoder_state --- next timestep decoder input state,prev_hybrid_attns-- alpha hybrid
        #decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs[0],emb_enc_inputs,emb_enc_inputs_fields)
        print ("!!!!!!!!!!!!")
        #print (len(decoder_outputs))
        ######################## equally paying attention to all the table content #####################################
        net_attn_score = tf.add_n(self.prev_hybrid_attn)
        _,batch_variance_score = tf.nn.moments(net_attn_score, axes= [1])
        batch_variance_loss = tf.reduce_mean(batch_variance_score,axis = 0)
        print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print (net_attn_score)
        print (batch_variance_score)
        print (batch_variance_loss)
        print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        ################################################################################################
      
      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [(2 * hps.hidden_dim), vsize_content_summary], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize_content_summary], dtype=tf.float32, initializer=self.trunc_norm_init)
        vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      final_dists = self._calc_final_dist(vocab_dists, self.prev_hybrid_attn)
      '''
      if FLAGS.pointer_gen:
        final_dists = self._calc_final_dist(vocab_dists, self.prev_hybrid_attn)
        print ("$$$$$$$$$$$$$$$$$$$$$$")
        print (final_dists)
      else: # final distribution is just vocabulary distribution
        final_dists = vocab_dists
      
        ## Add copy score to the original vocab_dists @@@@@@@@@@@@@@@@
        
        indices = self._enc_batch_content
        batch_indices_list = tf.unstack(indices,axis = 0)
        print (batch_indices_list,len(batch_indices_list))
        print (copy_scores[0], len(copy_scores))
        
        
        copy_dists = []
        for i,copy_score in enumerate(copy_scores):
          final_dists_decoder_step = []
          batch_copy_score_list = tf.unstack(copy_score,axis = 0)
          for j in range(len(batch_indices_list)):
            final_dist = tf.scatter_add(tf.Variable(lambda: tf.zeros(shape=(vsize_content_summary)), trainable=False,dtype=tf.float32),batch_indices_list[j],batch_copy_score_list[j])
            final_dists_decoder_step.append(final_dist)
          copy_dist_dec_step = tf.stack(final_dists_decoder_step,axis = 0)
          copy_dists.append(copy_dist_dec_step)
        final_dists = []
        for i in range(len(vocab_dists)):
          final_dist = tf.add(vocab_dists[0],copy_dists[0])
          final_dists.append(final_dist)

        final_dists = vocab_dists                             ##uncomment it, if we want to run with copynet @@@@@@@@@@@@@@
        '''
      ## Add predicted summary for the batch
      pred_batch_words_list = []
      for i,vocab_dist in enumerate(final_dists):
        vocab_dist = tf.argmax(vocab_dist,axis = 1)
        print ("^^^^^^^^^^^^^^^")
        print (vocab_dist)
        pred_batch_words_list.append(vocab_dist)
      enc_pred_words = tf.stack(pred_batch_words_list, axis=1,name ='stack')
      


      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          
          #if FLAGS.pointer_gen:
          # Calculate the loss per step
          # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
          loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
          batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
          for dec_step, dist in enumerate(final_dists):
            targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
            indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
            gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
            losses = -tf.log(gold_probs)
            loss_per_step.append(losses)

          # Apply dec_padding_mask and get loss
          self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask) + batch_variance_loss
          tf.summary.scalar('loss', self._loss)
          ### printing predicted summary and actual summary in the screen, one example for one batch...
          #example_number = random.randint(0,hps.batch_size-1)
          #enc_actual_summ = self._target_batch[example_number,:]
          enc_actual_summ = self._target_batch
          self._enc_actual_summ_ = enc_actual_summ
          #self._enc_pred_summ_ = enc_pred_words[example_number,:]
          self._enc_pred_summ_ = enc_pred_words
          #self.no_example = example_number
          ### end...
          '''
          else:        #baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss', self._loss)
          
          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)
          '''
    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)

    
  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    #loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    loss_to_minimize = self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
    # This class list the various methods that can be used to combine gradients in the graph.

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)
    
  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'enc_actual_summ' : self._enc_actual_summ_,
        'enc_pred_summ' : self._enc_pred_summ_,
        #'no_example' : self.no_example,
        'global_step': self.global_step,
    }
    '''
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    '''
    return sess.run(to_return, feed_dict)
    
  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_hybrid_attns, output_state):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)
    ###################################################################################convert list to a tensor
    prev_hybrid_attns = np.asarray(np.vstack(prev_hybrid_attns))
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print (type(batch.enc_batch_field))
    print (type(prev_hybrid_attns))
    print (prev_hybrid_attns)
    output_state = np.vstack(output_state)
    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
    
    feed = {
        self._enc_batch_field : batch.enc_batch_field,
        self._enc_batch_content : batch.enc_batch_content,
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
        self.prev_hybrid_attns : prev_hybrid_attns,
        self.output_state : output_state
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "prev_hybrid_attns": self.prev_hybrid_attn,
      "output_state" : self.output_state_h
    }

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    print ("%%%%%%%%%%%%%%%%%%%")
    
    results = sess.run(to_return, feed_dict=feed) # run the decoder step
    print (results['ids'])
    
    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(beam_size)]
    new_prev_hybrid_attns = results['prev_hybrid_attns'][0].tolist() ##list of list
    assert len(new_prev_hybrid_attns) == beam_size
    print (results['output_state'].shape)
    new_output_state = results['output_state'].tolist() ##list of list
    assert len(new_output_state) == beam_size
    print (len(new_prev_hybrid_attns[0]))
    '''
    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()
    '''
    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in xrange(beam_size)]
    '''
    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in xrange(beam_size)]
    '''
    return results['ids'], results['probs'], new_states, new_prev_hybrid_attns, new_output_state,p_gens
    

def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
