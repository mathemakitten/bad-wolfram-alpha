from preprocessing import char2idx

VOCAB_SIZE = len(char2idx)  # + 2  # +1 for the padding token, which is 0, and +1 for the delimiter newline/start token
QUESTION_MAX_LENGTH = 160
ANSWER_MAX_LENGTH = 30
