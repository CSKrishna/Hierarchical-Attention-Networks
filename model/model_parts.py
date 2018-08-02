import tensorflow as tf
import tensorflow.contrib.layers as layers

#wrapper for implementing a bidirectional GRU over word embedding sequences or sentence embedding sequences
class bidirectional_gru():    
    def __init__(self, cell_size):
        self.cell = tf.nn.rnn_cell.GRUCell(cell_size)
        
         
    def __call__(self, inputs, input_lengths, scope=None):
        with tf.variable_scope(scope or "bidirectionalGRU") as scope:
            (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, inputs, sequence_length = input_lengths, scope = scope, dtype=tf.float32)
        outputs = tf.concat((fw_outputs, bw_outputs), 2) 
        return outputs
    



def task_specific_attention(inputs, output_size, sequence_lengths,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).
    Args:
        inputs: Tensor of shape [batch_size, time_steps, input_size]           
            `time_steps` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension
    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2)       
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)                                        
        attention_weights = tf.nn.softmax(vector_attn, axis=1)
        attention_weights = attention_weights*mask
        norms = tf.reduce_sum(attention_weights, axis = 1, keepdims = True) + 1e-6     
        attention_weights = attention_weights / norms
        attention_weights = tf.expand_dims(attention_weights, axis = 2)        
        
        weighted_projection = inputs*attention_weights
        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs