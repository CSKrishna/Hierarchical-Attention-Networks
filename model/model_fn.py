"""Define the model."""

import tensorflow as tf
from model.model_parts import *
import tensorflow.contrib.layers as layers




def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)
tf
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    sentences = inputs['sentences']
    document_sizes = inputs['document_sizes']
    sentence_lengths = inputs['sentence_lengths']  
    is_training = (mode == 'train')
  

    if params.model_version == 'HAN':
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size], initializer = layers.xavier_initializer())
        sentence_embeddings =  tf.layers.dropout(tf.nn.embedding_lookup(embeddings, sentences),params.dropout_rate, training = is_training)
        
        sentence_embedder = bidirectional_gru(params.GRU_num_units)
        document_embedder = bidirectional_gru(params.GRU_num_units)
        
        #generate sentence level emebeddings
        shape = tf.shape(sentence_embeddings)       
        dim0  = shape[0]   #batch size - number of documentss
        dim1 = shape[1]   #number of padded sentences in a document
        dim2 = shape[2]       #sequence of words
        sentence_embeddings =  tf.reshape(sentence_embeddings, [dim0*dim1,dim2,params.embedding_size])
        sequence_lengths = tf.reshape(sentence_lengths, [-1]) 
        
        with tf.variable_scope('sentence_level') as scope:    
            outputs = sentence_embedder(sentence_embeddings, sequence_lengths, scope = scope)     
            
            #apply attention on  the outputs to get the final sentence embeddings  
            with tf.variable_scope('attention') as scope:
                sentence_level_outputs = task_specific_attention(outputs, 2*params.GRU_num_units, sequence_lengths,
            scope=scope)
                sentence_level_outputs = tf.layers.dropout(sentence_level_outputs,params.dropout_rate, training = is_training)  
        
        #generate document level embeddings
        dim1 = tf.shape(sentence_lengths)[1]
        dim0 = tf.shape(sentence_lengths)[0]
        sentence_level_outputs  = tf.reshape(sentence_level_outputs,[dim0, dim1, 2*params.GRU_num_units])      
        
        with tf.variable_scope('document_level') as scope:
            doc_outputs = document_embedder(sentence_level_outputs, document_sizes, scope = scope)             
         
            #apply attention on  the outputs to get the final sentence embeddings  
            with tf.variable_scope('attention') as scope:
                final_outputs = task_specific_attention(doc_outputs, 2*params.GRU_num_units, 
                                                        document_sizes, scope=scope)        
                final_outputs = tf.layers.dropout(final_outputs, params.dropout_rate, training = is_training)   
                
        # Compute logits from the output of the LSTM
        logits = tf.layers.dense(final_outputs, 10)

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
  

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(mode, inputs, params)
        predictions = tf.argmax(logits, -1)

    # Define loss and accuracy (we need to apply a mask to account for padding)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) 
    loss = tf.reduce_mean(losses)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
