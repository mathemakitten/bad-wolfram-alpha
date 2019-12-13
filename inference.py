import tensorflow as tf
import numpy as np
from config import *
from constants import ANSWER_MAX_LENGTH
#from lstm import EncoderDecoder, Encoder, Decoder, inference


# TODO port the real LSTM code

class Encoder(tf.keras.layers.Layer):

    def __init__(self, input_dim, embedding_size, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim, embedding_size, mask_zero=False)  # can't mask with cuda
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_state=True)  # return hidden states & sequences

    def call(self, x):
        input_mask = tf.dtypes.cast(tf.clip_by_value(x, 0, 1), dtype=tf.bool)
        input = self.embedding(x)
        output, hidden_state, cell_state = self.lstm(input, mask=input_mask)
        return [hidden_state, cell_state]


class Decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(output_dim, embedding_dim, mask_zero=False)
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_state=True, return_sequences=True)   # apply data at each timestep?
        self.decoder_output = tf.keras.layers.Dense(output_dim)

    def call(self, x, encoder_states):
        # input_mask = tf.dtypes.cast(tf.clip_by_value(tf.expand_dims(x, axis=-1), 0, 1), tf.float32)
        x = self.embedding(x)
        # x = input_mask * x
        x, hidden_state, cell_state = self.lstm(inputs=x, initial_state=encoder_states)
        y = self.decoder_output(x)

        return y, [hidden_state, cell_state]


class EncoderDecoder(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, max_len):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim, hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, output_dim)

        self.max_length = max_len
        self.output_dim = output_dim

    def call(self, inputs, targets=None):  # targets = None if inference mode

        outputs = tf.zeros((inputs.shape[0], 0, self.output_dim), dtype=tf.float32)
        output_tokens = []
        batch_size = inputs.shape[0]
        len_target_sequences = self.max_length

        decoder_states = self.encoder(inputs)  # initialize decoder lstm states with encoder states

        decoder_output_token = tf.ones((batch_size, 1))  # start token (1) for inference
        for timestep in range(len_target_sequences):
            if targets is not None:  # train
                decoder_output, decoder_states = self.decoder(tf.expand_dims(targets[:, timestep], axis=1), decoder_states)
            else:  # inference
                decoder_output, decoder_states = self.decoder(decoder_output_token, decoder_states)
            decoder_output_token = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)
            output_tokens.append(decoder_output_token)
            outputs = tf.concat([outputs, decoder_output], axis=1)

        return outputs, output_tokens

# load pre-padded data
questions_encoded = np.array(np.load('cache/questions_encoded_padded_interpolate_arithmetic__add_or_sub.npy'))
answers_encoded = np.array(np.load('cache/answers_encoded_padded_interpolate_arithmetic__add_or_sub.npy'))

# TODO preprocess into single arrays
dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
input_data = dataset.take(1).batch(1)

model = EncoderDecoder(input_dim=VOCAB_SIZE, embedding_dim=EMBEDDING_SIZE, hidden_dim=LSTM_HIDDEN_SIZE, output_dim=VOCAB_SIZE, max_len=ANSWER_MAX_LENGTH)
# model.save_weights('experiment_results/test')
model.load_weights('experiment_results/2019_12_12_00:38-easy-arithmetic__add_or_sub-fullrun_batch512_lstm1024_15epochs/model_weights')
inference(input_data, model)