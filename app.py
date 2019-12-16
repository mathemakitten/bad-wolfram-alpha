from flask import Flask, render_template, url_for, request


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    import tensorflow as tf
    from lstm import EncoderDecoder, Encoder, Decoder, inference
    import pickle
    import numpy as np

    char2idx = pickle.load(open('char2idx.pkl', 'rb'))
    idx2char = pickle.load(open('idx2char.pkl', 'rb'))

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

    def inference(question):

        VOCAB_SIZE = len(char2idx)

        ANSWER_MAX_LENGTH = 30
        EMBEDDING_SIZE = 512
        LSTM_HIDDEN_SIZE = 512

        questions_encoded = [char2idx[q] for q in question]
        answers_encoded = [char2idx[q] for q in question]

        questions_encoded = np.expand_dims(np.array(questions_encoded), axis=0)

        # TODO preprocess into single arrays
        #dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
        #input_data = dataset.take(1).batch(1)

        #data = input_data[0]

        model = EncoderDecoder(input_dim=VOCAB_SIZE, embedding_dim=EMBEDDING_SIZE, hidden_dim=LSTM_HIDDEN_SIZE,
                               output_dim=VOCAB_SIZE, max_len=ANSWER_MAX_LENGTH)
        # model.save_weights('experiment_results/test')
        model.load_weights('model_weights')

        print(questions_encoded)
        outputs, output_tokens = model(questions_encoded)

        predicted_text = token_to_text(output_to_tensor(output_tokens))[0]
        first_stop_token = predicted_text.index('~')
        predicted_text = predicted_text[0:first_stop_token]

        return predicted_text

    if request.method == 'POST':
        message = request.form['message']
        #data = [message]
        data = message
        my_prediction = inference(data)
        print("MY PREDICTION IS: {}".format(my_prediction))
        #my_prediction = 0 #inference(data)

    return render_template('result.html', prediction=my_prediction, input=message)


if __name__ == '__main__':
    app.run(debug=False)
