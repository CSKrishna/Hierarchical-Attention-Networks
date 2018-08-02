"""Create the input data pipeline using `tf.data`"""

import pandas as pd
import tensorflow as tf


def load_dataset_from_text(path_dataset):
    """Create tf.data Instance from python pickle file

    Args:
        path_txt: (string) path to pickle file
   
    Returns:
        dataset: (tf.Dataset) yielding a document, label pair
    """
    # Load and pre prepare the dataset    
    data_train = pd.read_pickle(path_dataset)
    
    docs = data_train[["Sent"]].values.tolist()
    docs = [doc[0] for doc in docs]
    
    labels = data_train[["rating"]]
    labels = labels.values.tolist()
    labels = [label[0] for label in labels]
    
    def gen():
        for label, doc in zip(labels, docs):
            yield label, doc
            
    ds = tf.data.Dataset.from_generator(gen, (tf.int32, tf.string), ([], [None]))
 
    return ds


def input_fn(mode, ds, vocab, params):
    """Input function HAN

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        ds: tf.data instance where each element comrrises a document, label tuple
        vocab: pvocab: (tf.lookuptable)
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1
           
    def _read_py_function(label, sentences):     
        doc_len = len(sentences)
        label = label - 1
        sen_len = [len(str(sentence).split(" ")) for sentence in sentences]  
        return label, doc_len, sen_len, sentences
    
    ds = ds.map(lambda label, sentences : tuple(tf.py_func(
     _read_py_function, [label, sentences], [tf.int32, tf.int32, tf.int32, tf.string])), num_parallel_calls=4)
    
    
    def transform(doc, default_value='<pad>'):      
        # Split sentence
        out = tf.string_split(doc) 
        
        # Convert to Dense tensor, filling with default value
        out = tf.sparse_tensor_to_dense(out, default_value=default_value)   
    
        out = vocab.lookup(out)
        out = tf.cast(out, tf.int32)
        return out
    
    ds= ds.map(lambda label, size1, size2, doc : (label, size1, size2, transform(doc)), num_parallel_calls=4)

    # Create batches and pad the sentences of different length
    padded_shapes = (tf.TensorShape([]),
                     tf.TensorShape([]),   # doc of unknown size
                     tf.TensorShape([None]),  # sentence lenghts
                     tf.TensorShape([None, None])) # sentence tokens
    padding_values = (0,0,0,0)                  

    ds = (ds
        .padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        .prefetch(1))  # make sure you always have one batch ready to serve          

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = ds.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    labels, document_sizes, sentence_lengths, sentences =  iterator.get_next()
    init_op = iterator.initializer    
    inputs = {
        'labels': labels,
        'document_sizes': document_sizes,
        'sentence_lengths': sentence_lengths,
        'sentences': sentences,
        'iterator_init_op': init_op
       }

    return inputs
