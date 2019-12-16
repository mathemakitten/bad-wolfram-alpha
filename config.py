import os
#from constants import VOCAB_SIZE
from datetime import datetime as dt
current_time = dt.now().strftime('%Y%m%d_%H_%M-')

BATCH_SIZE = 128
EMBEDDING_SIZE = 512
LSTM_HIDDEN_SIZE = 512
NUM_EPOCHS = 50
NUM_EXAMPLES = 666666*3
p_test = .2
