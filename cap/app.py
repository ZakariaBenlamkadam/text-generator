# Import the necessary libraries
from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pickle

# Initialize Flask app
app = Flask(__name__)

# Define sequence length and GPU availability
sequence_length = 100
train_on_gpu = torch.cuda.is_available()

# Function to load preprocessed data
def load_preprocess(file_path='./preprocess.p'):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Load preprocessed data
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()

# Ensure the token_dict includes the PADDING token
def ensure_padding_token(token_dict):
    if 'PADDING' not in token_dict:
        token_dict['PADDING'] = '<PAD>'
    return token_dict

token_dict = ensure_padding_token(token_dict)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_size)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, hidden):
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        out = self.fc(lstm_out)
        out = out.view(x.size(0), -1, len(vocab_to_int))
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if train_on_gpu:
            hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        return hidden

# Load the trained model
def load_model(file_path, model):
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
    model.eval()

# Generate text with temperature and top-k sampling
def generate(rnn, prime_ids, int_to_vocab, token_dict, pad_value, text_length=100, temperature=1.0, top_k=5):
    rnn.eval()
    current_seq = np.full((1, sequence_length), pad_value)
    prime_len = len(prime_ids)
    current_seq[0, -prime_len:] = prime_ids
    predicted = [int_to_vocab[idx] for idx in prime_ids]
    hidden = rnn.init_hidden(1)

    for _ in range(text_length):  # Use text_length instead of predict_len
        current_seq = torch.LongTensor(current_seq).to(device)
        output, hidden = rnn(current_seq, hidden)
        output = output.div(temperature).exp()
        p, top_i = output.topk(top_k)
        top_i = top_i.numpy().squeeze()
        p = p.detach().numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())
        word = int_to_vocab[word_i]
        predicted.append(word)
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[0, -1] = word_i

    gen_sentences = ' '.join(predicted)
    for key, token in token_dict.items():
        gen_sentences = gen_sentences.replace(f' {token.lower()} ', key)
    return gen_sentences

# Post-process the generated text
def post_process_generated_text(generated_script, token_replacements):
    for token, char in token_replacements.items():
        generated_script = generated_script.replace(token, char)
    generated_script = ' '.join(generated_script.split())
    return generated_script

# Define hyperparameters
vocab_size = len(vocab_to_int)
output_size = vocab_size
embedding_dim = 200
hidden_dim = 250
n_layers = 2
dropout = 0.5

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the RNN model
rnn_loaded = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout)
if train_on_gpu:
    rnn_loaded.cuda()

# Load the trained model
load_model('./trained_rnn2.pth', rnn_loaded)
print('Model Loaded')

# Serve the HTML file
@app.route('/')
def index():
    with open('templates/index.html', 'r') as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    prime_phrase = request.json['prime_phrase']
    text_length = int(request.json['text_length'])  # Get the text length from the request
    pad_word = token_dict['PADDING']

    prime_ids = [vocab_to_int[word] for word in prime_phrase.split() if word in vocab_to_int]
    generated_script = generate(rnn_loaded, prime_ids, int_to_vocab, token_dict, vocab_to_int[pad_word], text_length)  # Pass text_length to the generate function

    # Post-process the generated text
    token_replacements = {
        '||period||': '.',
        '||comma||': ',',
        '||quotation_mark||': '"',
        '||semicolon||': ';',
        '||exclamation_mark||': '!',
        '||question_mark||': '?',
        '||left_parenthesis||': '(',
        '||right_parenthesis||': ')',
        '||dash||': '--',
        '||return||': ' '
    }

    generated_script = post_process_generated_text(generated_script, token_replacements)
    return jsonify({'generated_text': generated_script})

if __name__ == '__main__':
    app.run(debug=True)

