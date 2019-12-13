""" LSTM Encoder-Decoder architecture with the Keras imperative API.  """
# https://github.com/yusugomori/deeplearning-tf2/blob/master/models/encoder_decoder_lstm.py

import tensorflow as tf
import numpy as np
import os
import argparse
from constants import ANSWER_MAX_LENGTH
from preprocessing import idx2char  # TODO cache questions/answers_encoded as .npy files
from config import *
from utils import get_logger
import time

tb_logdir = os.path.join(EXPERIMENT_DIR, 'tensorboard')
logger = get_logger('validation_log')
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eager', metavar='eager_mode', type=bool, default=True, help='Eager mode on, else Autograph')
parser.add_argument('--gpu_id', metavar='gpu_id', type=str, default="1", help='The selected GPU to use, default 1')
args = parser.parse_args()

tf.config.experimental_run_functions_eagerly(args.eager)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# TODO ADD SLACKBOT, get Ray's code
# TODO: Take functions out of main

# load pre-padded data
# questions_encoded = np.array(np.load('cache/questions_encoded_padded.npy'))
# answers_encoded = np.array(np.load('cache/answers_encoded_padded.npy'))
#questions_encoded = np.array(np.load('cache/questions_encoded_padded_interpolate_arithmetic__add_or_sub.npy'))
#answers_encoded = np.array(np.load('cache/answers_encoded_padded_interpolate_arithmetic__add_or_sub.npy'))
questions_encoded = np.array(np.load('cache/questions_encoded_padded_interpolate_arithmetic__add_or_sub_ALL_DIFFICULTY.npy'))
answers_encoded = np.array(np.load('cache/answers_encoded_padded_interpolate_arithmetic__add_or_sub_ALL_DIFFICULTY.npy'))

dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
input_data = dataset.take(NUM_EXAMPLES).shuffle(questions_encoded.shape[0]).batch(BATCH_SIZE) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
NUM_TRAINING_BATCHES = int(NUM_EXAMPLES/BATCH_SIZE*(1-p_test))
train_data = input_data.take(NUM_TRAINING_BATCHES).repeat(NUM_EPOCHS)
valid_data = input_data.skip(NUM_TRAINING_BATCHES)

# #  load data
# questions_encoded = np.array(np.load('cache/questions_encoded.npy', allow_pickle=True))
# answers_encoded = np.array(np.load('cache/answers_encoded.npy', allow_pickle=True))
# questions_tensor = tf.ragged.constant(questions_encoded)
# answers_tensor = tf.ragged.constant(answers_encoded)
# dataset = tf.data.Dataset.from_tensor_slices((questions_tensor, answers_tensor))
# input_data = dataset.take(TRAINING_EXAMPLES).shuffle(questions_encoded.shape[0]).repeat(NUM_EPOCHS)\
#              .padded_batch(BATCH_SIZE, padded_shapes=([None,], [None,]))\
#              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


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

def get_accuracy(output_tokens, targets):
    correct = 0
    # targets = token_to_text(targets)
    # predictions = token_to_text(output_to_tensor(output_tokens))
    output_tokens = output_to_tensor(output_tokens)
    for output, target in zip(output_tokens, targets[:, 1:]):
        target = target.numpy().tolist()
        output = output.numpy().tolist()
        stop_index = target.index(2)
        if output[:stop_index] == target[:stop_index]:  # index 2 is stop token
            correct += 1

    return correct/len(output_tokens)

# TODO HN: fix CUPTI on Quoc so we can see the graph visualization
def tensorboard_profile(writer, logdir):
    with writer.as_default():
        tf.summary.trace_export(name="Trace_loss", step=0, profiler_outdir=logdir)

@tf.function
def train_step(inputs, targets, model):
    with tf.GradientTape() as tape:
        # targets[:, :-1] to limit output to 30 chars from 31
        outputs, _ = model(inputs[:, :], targets[:, :-1])  # softmax outputs
        loss_mask = tf.dtypes.cast(tf.clip_by_value(targets[:, 1:], 0, 1), tf.float32)
        # targets[:, 1:] to remove start token so model predicts target's actual chars
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets[:, 1:], outputs, from_logits=True)
        loss = loss * loss_mask
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [(tf.clip_by_value(grad, -0.1, 0.1)) for grad in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Get gradient updates
    updates = [tf.reduce_mean(tf.abs(g)) / tf.reduce_mean(tf.abs(v)) if g is not None else None for g, v in
               zip(grads, model.trainable_variables)]
    updates = {v.name: u for v, u in zip(model.trainable_variables, updates) if u is not None}

    loss = tf.reduce_sum(loss)/tf.reduce_sum(loss_mask)
    return loss, updates

def get_validation_metrics(validation_data, model):
    loss_list = []
    accuracy_list = []
    for i, data in enumerate(validation_data):
        inputs = data[0]
        targets = data[1]

        outputs, output_tokens = inference_step(inputs[:, :], model)
        loss_mask = tf.dtypes.cast(tf.clip_by_value(targets[:, 1:], 0, 1), tf.float32)
        validation_loss = tf.keras.losses.sparse_categorical_crossentropy(targets[:, 1:], outputs, from_logits=True)
        validation_loss = validation_loss * loss_mask
        accuracy = get_accuracy(output_tokens, targets)

        inputs = token_to_text(inputs[:, :])
        targets = token_to_text(targets)
        predictions = token_to_text(output_to_tensor(output_tokens))

        if i % 100 == 0:  # Print some examples from one batch # TODO should this be % 100?
            for sample_index in range(3):
                logger.info(f'Input: {inputs[sample_index]}')
                logger.info(f'Target: {targets[sample_index]}')
                logger.info(f'Prediction: {predictions[sample_index]} \n')

        accuracy_list.append(accuracy)
        loss_list.append(tf.reduce_sum(validation_loss)/tf.reduce_sum(loss_mask).numpy())

    mean_loss = np.mean(loss_list)
    mean_acc = np.mean(accuracy_list)
    logger.info(f'Validation loss: {mean_loss}')
    logger.info(f'Validation accuracy: {mean_acc}')
    return mean_loss, mean_acc

def train(training_data, model):
    valid_loss_list = []  # for early stopping
    best_loss = np.inf  # for model check-pointing

    # Setup training tracking capabilities
    progress_bar = tf.keras.utils.Progbar(int(NUM_TRAINING_BATCHES * NUM_EPOCHS), verbose=1)
    writer = tf.summary.create_file_writer(tb_logdir)

    for i, data in enumerate(training_data):
        inputs = data[0]
        targets = data[1]
        progress_bar.update(i)

        if i == 0:
            tf.summary.trace_on(graph=True, profiler=True)

        # Run a single training batch
        start_time = time.time()
        loss, updates = train_step(inputs, targets, model)
        time_per_batch = round(time.time() - start_time, 2)

        if i == 0:
            tensorboard_profile(writer, tb_logdir)
            tf.summary.trace_off()

        if i % 10 == 0:
            print(f' Train loss at batch {i}: {loss} - Time per batch: {time_per_batch}')
            with writer.as_default():
                tf.summary.scalar(f"Losses/total_loss", loss, step=i)

                for variable in model.trainable_variables:
                    tf.summary.histogram("Weights/{}".format(variable.name), variable, step=i)

                for layer, update in updates.items():
                    tf.summary.scalar("Updates/{}".format(layer), update, step=i)

                mean_updates = tf.reduce_mean(list(updates.values()))
                max_updates = tf.reduce_max(list(updates.values()))
                tf.summary.scalar("Mean_Max_Updates/Mean_updates", mean_updates, step=i)
                tf.summary.scalar("Mean_Max_Updates/Max_updates", max_updates, step=i)

                writer.flush()

        if i % 20 == 0:
            valid_loss, valid_acc = get_validation_metrics(valid_data, model)
            valid_loss_list.append(valid_loss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                logger.info(f'Saving on batch {i}')
                logger.info(f'New best validation loss: {best_loss}')
                model.save_weights(os.path.join(EXPERIMENT_DIR, 'model_weights'), save_format='tf')
            # early stopping
            print(valid_loss_list)
            if all([valid_loss < best_loss for valid_loss in valid_loss_list[-5:]]):
                return

def output_to_tensor(tokens):
    tensor_tokens = tf.squeeze(tf.convert_to_tensor(tokens), axis=2)
    return tf.transpose(tensor_tokens)

def token_to_text(batch_tensor):
    batch_array = batch_tensor.numpy()
    text_outputs = []
    for sequence_pred in batch_array:
        text = ''.join([idx2char[pred] for pred in sequence_pred])
        text_outputs.append(text)
    return text_outputs

def inference_step(inputs, model):
    outputs, output_tokens = model(inputs)
    return outputs, output_tokens

def inference(inference_data, model):
    accuracy_list = []
    validation_loss_list = []
    for i, data in enumerate(inference_data):
        inputs = data[0]
        targets = data[1]

        outputs, output_tokens = inference_step(inputs[:, :], model)
        loss_mask = tf.dtypes.cast(tf.clip_by_value(targets[:, 1:], 0, 1), tf.float32)
        validation_loss = tf.keras.losses.sparse_categorical_crossentropy(targets[:, 1:], outputs, from_logits=True)
        validation_loss = validation_loss * loss_mask
        accuracy = get_accuracy(output_tokens, targets)
        inputs = token_to_text(inputs[:, :])
        targets = token_to_text(targets)
        predictions = token_to_text(output_to_tensor(output_tokens))
        for sample_index in range(len(inputs)):
            print(f'Input: {inputs[sample_index]}')
            print(f'Target: {targets[sample_index]}')
            print(f'Prediction: {predictions[sample_index]} \n')
        accuracy_list.append(accuracy)
        validation_loss_list.append(tf.reduce_sum(validation_loss)/tf.reduce_sum(loss_mask).numpy())
    print(f'Validation Accuracy: {np.mean(accuracy_list)}')
    print(f'Validation Loss: {np.mean(validation_loss_list)}')

if __name__ == '__main__':  # TODO HN move these function definitions out of main... hahahahaha yikes
    np.random.seed(1234)
    tf.random.set_seed(1234)

    model = EncoderDecoder(input_dim=VOCAB_SIZE, embedding_dim=EMBEDDING_SIZE, hidden_dim=LSTM_HIDDEN_SIZE, output_dim=VOCAB_SIZE, max_len=ANSWER_MAX_LENGTH)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    tf.keras.utils.Progbar

    logger.info("Logging to {}".format(EXPERIMENT_DIR))

    train(train_data, model)
    #model.load_weights('experiment_results/2019_12_12_00:38-easy-arithmetic__add_or_sub-fullrun_batch512_lstm1024_15epochs/model_weights')
    inference(valid_data, model)