from flask import Flask, render_template, url_for, request


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    '''
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    '''

    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)


    def inference(question):
        import tensorflow as tf
        from lstm import EncoderDecoder, Encoder, Decoder, inference
        import pickle

        idx2char = pickle.load(open('idx2char.pkl', 'rb'))
        char2idx = pickle.load(open('char2idx.pkl', 'rb'))

        VOCAB_SIZE = len(char2idx) + 2

        ANSWER_MAX_LENGTH=30
        BATCH_SIZE = 128
        EMBEDDING_SIZE = 512
        LSTM_HIDDEN_SIZE = 1024
        NUM_EPOCHS = 50
        NUM_EXAMPLES = 666666 * 3
        p_test = .2

        questions_encoded = [char2idx[q] for q in question]
        answers_encoded = [char2idx[q] for q in question]

        # TODO preprocess into single arrays
        dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
        input_data = dataset.take(1).batch(1)

        model = EncoderDecoder(input_dim=VOCAB_SIZE, embedding_dim=EMBEDDING_SIZE, hidden_dim=LSTM_HIDDEN_SIZE,
                               output_dim=VOCAB_SIZE, max_len=ANSWER_MAX_LENGTH)
        # model.save_weights('experiment_results/test')
        model.load_weights('model_weights')

        outputs, output_tokens = model(question)

        return outputs, output_tokens

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = 0 #inference(data)

    return render_template('result.html', prediction=my_prediction, input=message)


if __name__ == '__main__':
    app.run(debug=True)