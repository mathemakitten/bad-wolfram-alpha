from flask import Flask, render_template, url_for, request, send_from_directory
import os

app = Flask(__name__)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


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

        questions_encoded = np.expand_dims(np.array(questions_encoded), axis=0)

        model = EncoderDecoder(input_dim=VOCAB_SIZE, embedding_dim=EMBEDDING_SIZE, hidden_dim=LSTM_HIDDEN_SIZE,
                               output_dim=VOCAB_SIZE, max_len=ANSWER_MAX_LENGTH)
        model.load_weights('20191215_21_03-helen_all_add_subtract/model_weights')

        print(questions_encoded)
        outputs, output_tokens = model(questions_encoded)

        predicted_text = token_to_text(output_to_tensor(output_tokens))[0]
        first_stop_token = predicted_text.index('~')
        predicted_text = predicted_text[0:first_stop_token]

        return predicted_text

    if request.method == 'POST':
        message = request.form['message']

        message = message.strip()

        for letter in message:  # remove disallowed characters
            if letter not in char2idx.keys():
                message = message.replace(letter, '')

        data = message
        my_prediction = inference(data)
        print("MY PREDICTION IS: {}".format(my_prediction))
        # TODO strip bad characters before model inference
        #my_prediction = 0 #inference(data)

    return render_template('result.html', prediction=my_prediction, input=message)


if __name__ == '__main__':

    # server side:
    app.run_server(host='0.0.0.0', port=5000, debug=False)

    # local:
    # app.run(host='0.0.0.0', port=5000, debug=False)

