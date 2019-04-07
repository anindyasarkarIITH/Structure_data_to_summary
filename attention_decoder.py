
"""This file defines the decoder and Dispatcher"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops


def attention_decoder(decoder_inputs,emb_enc_inputs,emb_enc_inputs_fields, initial_state, encoder_states, enc_padding_mask, cell,dec_lstm_hidden_dim,vsize_field,enc_batch_field_link,cell_output, initial_state_attention=False,prev_hybrid_attn = None,pointer_gen=True):
  
  with variable_scope.variable_scope("attention_decoder") as scope:
    print ("!!!!")
    print (enc_batch_field_link)
    batch_size = encoder_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    dispatcher_attention_size = encoder_states.get_shape()[2].value # if this line fails, it's because the attention size isn't defined
    emb_dec_size = decoder_inputs[0].get_shape()[1].value
    score_layer_hidden_dim = (2 * dec_lstm_hidden_dim)
    ## shape required for calculate hybrid attention
    hybrid_gate_mat_shape = (2 * (emb_enc_inputs_fields.get_shape()[2].value ) + dec_lstm_hidden_dim)
    dispatcher_output = array_ops.zeros([batch_size, dispatcher_attention_size]) # set to zero for initial iteration of decoder
    #in the literature dispatcher_output is denoted as "at"
    decoder_output_prev = decoder_inputs[0] # set to <start> token for initial iteration of decoder. from the next iteration it will be the output to previous decoder step.in the literature it is denoted as "yt-1".
    decoder_state = initial_state # last encoder state output
    max_encoder_length = tf.shape(encoder_states)[1]
    if (initial_state_attention == False):
      prev_hybrid_attn = array_ops.zeros([batch_size,max_encoder_length]) # set to zero initially for initial iteration of decoder. from the next iteration it will be calculated accordingly.in the paper it is denoted as "alpha_t-1_hybrid"
      cell_output = array_ops.zeros([batch_size, dec_lstm_hidden_dim])

    def masked_attention(e):
      """Take softmax of e then apply enc_padding_mask and re-normalize"""
      attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
      attn_dist *= enc_padding_mask # apply mask
      masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
      return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize

    outputs = []
    copy_scores = []
    dispatcher_outputs = []
    p_gens = []
    prev_hybrid_attns = []
    # Define the weights and bias to implement equation 13
    with variable_scope.variable_scope("x_t_gate_matrix"):
        w_d = variable_scope.get_variable("w_d", [(dispatcher_attention_size + emb_dec_size), emb_dec_size])
        b_d = variable_scope.get_variable("b_d",[emb_dec_size], initializer=tf.constant_initializer(0.0))

    # Define the weights and bias to implement equation 13
    with variable_scope.variable_scope("score_function_matrix"):
        w_s = variable_scope.get_variable("w_s", [dec_lstm_hidden_dim, score_layer_hidden_dim])
        b_s = variable_scope.get_variable("b_s",[score_layer_hidden_dim], initializer=tf.constant_initializer(0.0))


    ###### Implementation of dispatcher attention module
    # Define the weights and biases to implement content based attention
    with variable_scope.variable_scope("content_based_attention_matrix"):
        w_f = variable_scope.get_variable("w_f", [emb_dec_size, emb_dec_size])
        b_f = variable_scope.get_variable("b_f",[emb_dec_size], initializer=tf.constant_initializer(0.0))
        w_c = variable_scope.get_variable("w_c", [emb_dec_size, dec_lstm_hidden_dim])
        b_c = variable_scope.get_variable("b_c",[dec_lstm_hidden_dim], initializer=tf.constant_initializer(0.0))
    # Define the weight to implement link based attention
    with variable_scope.variable_scope("link_based_attention_matrix"):
        w_L = variable_scope.get_variable("w_L", [vsize_field, vsize_field])
        #link_mat = variable_scope.get_variable("link_matrix", [422, 422])
        

    with variable_scope.variable_scope("hybrid_gate_matrix"):
        w_T = variable_scope.get_variable("w_T", [hybrid_gate_mat_shape, 1])

    with variable_scope.variable_scope("copy_score_matrix"):
        w_copy = variable_scope.get_variable("w_copy",[batch_size,dispatcher_attention_size,dispatcher_attention_size])
    print (w_copy)

    def dispatcher_attention_module(emb_enc_inputs_fields,encoder_states,prev_hybrid_attn,inp,cell_output):
      ''' agrs: emb_enc_inputs_fields : denoted as f in the paper
                encoder_states : denoted as h in the paper
                inp : denoted as yt-1 in the paper while training
                prev_hybrid_attn : denoted as "alpha_t-1_hybrid" in the paper
                cell_output : denoted as h_prime_t-1 in the paper'''
      # Implementation of content based attention as mentioned in the paper
      # Implementation of equation 5 in the paper
      field_attn_content_based = tf.matmul(inp,w_f) + b_f
      field_attn_content_based = tf.expand_dims(field_attn_content_based, axis=-1)
      field_attn_content_based_final = tf.matmul(emb_enc_inputs_fields, field_attn_content_based)
      field_attn_content_based_final = tf.reduce_sum(field_attn_content_based_final, axis=2)
      # Implementation of equation 6 in the paper
      content_attn_content_based = tf.matmul(inp,w_c) + b_c
      content_attn_content_based = tf.expand_dims(content_attn_content_based, axis=-1)
      content_attn_content_based_final = tf.matmul(encoder_states, content_attn_content_based)
      content_attn_content_based_final = tf.reduce_sum(content_attn_content_based_final, axis=2)
      ### Implementation of equation 7 in the paper
      content_based_attn = tf.nn.softmax(field_attn_content_based_final * content_attn_content_based_final, name="softmax")
      content_based_attn = masked_attention(content_based_attn) # masked attention is required because of padding
      #print (content_based_attn)
      # Implementation equation 8 and 9 for Link based attention
      # carve out only the relevant values from the Link matrix
      relevant_field_mat = tf.nn.embedding_lookup(w_L, enc_batch_field_link)
      print ("&&&&&&&&&&&&&&&")
      print (relevant_field_mat)
      relevant_field_mat_final = tf.map_fn(lambda u: tf.gather(u[0],u[1],axis=1),[relevant_field_mat, enc_batch_field_link], dtype=relevant_field_mat.dtype)
      ## Implementation of equation 9 for link based attention
      link_based_attn = tf.nn.softmax(tf.reduce_sum(tf.expand_dims(prev_hybrid_attn, axis = -1) * relevant_field_mat_final, axis=1),name="softmax")
      link_based_attn = masked_attention(link_based_attn) # masked attention is required because of padding
      #print (link_based_attn)
      ## Implementation of hybrid attention..equation 10 and 11 as described in the paper
      # calculation of e_t
      field_weight_e_t = tf.expand_dims(link_based_attn, axis=-1)
      #print (field_weight_e_t)
      e_t_timestack = tf.multiply(field_weight_e_t, emb_enc_inputs_fields)
      e_t = tf.reduce_sum(e_t_timestack, axis=1)
      print (e_t) #(2,128)
      print (inp) #(2,128)
      print (cell_output) #(2,256)
      hybrid_gate = tf.concat([cell_output, e_t, inp], axis=-1)
      hybrid_gate_final = tf.nn.sigmoid(tf.matmul(hybrid_gate, w_T))
      hybrid_gate_tilde = (0.2 * hybrid_gate_final) + 0.5
      #print (hybrid_gate_tilde)
      ## Implementation of equation 11 as described in the paper
      content_hybrid_attn_final = tf.multiply(hybrid_gate_tilde,content_based_attn)
      hybrid_gate_tilde_link = (1 - hybrid_gate_tilde)
      field_hybrid_attn_final = tf.multiply(hybrid_gate_tilde_link,link_based_attn)
      final_hybrid_attn = content_hybrid_attn_final + field_hybrid_attn_final
      #print (final_hybrid_attn)
      prev_hybrid_attn = final_hybrid_attn #######time rolling...
      # Implementation of equation 12 as described in the paper
      #print (encoder_states)
      #print (final_hybrid_attn)
      final_hybrid_attn_expanded = tf.expand_dims(final_hybrid_attn, axis=-1)
      dispatcher_output_timestack = tf.multiply(final_hybrid_attn_expanded,encoder_states)
      dispatcher_output = tf.reduce_sum(dispatcher_output_timestack, axis=1)
      #print (dispatcher_output)
      #pass
      # dispatcher output is denoted as at in the paper
      return (dispatcher_output,prev_hybrid_attn)

    #######
    ############# This function is required to calculate pointer generation probability
    
    def linear(args, output_size, bias, bias_start=0.0, scope=None):
      """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

      Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
      """
  
      if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
      if not isinstance(args, (list, tuple)):
        args = [args]

      # Calculate the total size of arguments on dimension 1.
  
      total_arg_size = 0
      shapes = [a.get_shape().as_list() for a in args]
      for shape in shapes:
        if len(shape) != 2:
          raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
          raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
          total_arg_size += shape[1]
  
      # Now the computation.
      with tf.variable_scope(scope or "Linear"):
    
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
          res = tf.matmul(args[0], matrix)
        else:
          res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
          return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
      return res + bias_term

    ###############
    ###### DECODER TIME ROLL COMPUTATION
    for i, inp in enumerate(decoder_inputs):
      tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      dispatcher_output, prev_hybrid_attn = dispatcher_attention_module(emb_enc_inputs_fields,encoder_states,prev_hybrid_attn,inp,cell_output)
        #dispatcher_output, prev_hybrid_attn = dispatcher_attention_module(emb_enc_inputs_fields,encoder_states,prev_hybrid_attn,inp)
      #print ("@@@@@@@@@@@@@@")
      #print (prev_hybrid_attn)
      #print ("@@@@@@@@@@@@@@@")
      #### Implement equation 13 as in the paper
      x_t = tf.tanh(tf.matmul(tf.concat([dispatcher_output, inp], axis=-1), w_d) + b_d)  
      #print (x_t)
      ### Implement decoder LSTM, we need cell_output.. in the paper cell_output is defined by h_prime_t..used by equation 14
      cell_output, state = cell(x_t, decoder_state) 
      decoder_state = state
      #print (cell_output)
      ############## ADD POINTER GENERATION PART###############
      # Dispatcher_output is same as context_vector...
      # Calculate p_gen
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          p_gen = linear([dispatcher_output, state.c, state.h, x_t], 1, True) # Tensor shape (batch_size, 1)
          print ("@@@@@@@@@@")
          print (p_gen)
          print ("@@@@@@@@@@")
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)
      #######################################################
      # Implementation of copy score equation as described in  equation 15 in the paper
      copy_score_part = tf.nn.sigmoid(tf.matmul(encoder_states,w_copy))
      #print (copy_score_part)
      cell_output_expanded = tf.expand_dims(cell_output,axis = -1)
      copy_score = tf.reduce_sum(tf.matmul(copy_score_part,cell_output_expanded),axis = -1)
      copy_score = masked_attention(copy_score)
      # Implement equation 14 as in the paper
      s_t = tf.matmul(cell_output, w_s) + b_s
      #### for copynet compute s_copy and add with s_t..where both are same shape..
      copy_scores.append(copy_score)
      outputs.append(s_t)
      dispatcher_outputs.append(dispatcher_output)
      prev_hybrid_attns.append(prev_hybrid_attn)
    return (outputs,copy_scores,prev_hybrid_attns,cell_output,decoder_state,p_gens)




    
    
