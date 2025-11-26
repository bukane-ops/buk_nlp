# necessary libraries
import tensorflow as tf # deep learning framework to train the encoder-decoder translation model and build the LSTM layers
import numpy as np # creating padded sequences
from collections import Counter # to build word-frequency vocabularies
import random # to shuffle or sample the data

# Check TensorFlow and Keras versions for compatibility
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
if not tf.__version__.startswith('2'):
    raise ValueError("This code requires TensorFlow 2.x")

# main translation program
# Step 1: Prepare the dataset
data = [
    ("I am happy", "Inu mi dùn"),
    ("You are sad", "O ní ìbànújẹ"),
    ("She is tired", "Ó rẹ̀ mi"),
    ("We are hungry", "A ní ebi"),
    ("He is angry", "Ó bínú"),
    ("They are busy", "Wọ́n ń ṣiṣẹ́"),
    ("I am cold", "Mo tutù"),
    ("You are late", "O pé"),
    ("She is happy", "Inu rẹ̀ dùn"),
    ("We are ready", "A ti ṣetan")
]

# Build vocabularies
def build_vocab(sentences, lang):
    tokens = Counter() # creates a special dictionary for counting words
    for sent in sentences:
        tokens.update(sent.split())
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for i, token in enumerate(tokens.keys(), 3):
        vocab[token] = i
    return vocab

# extract English and Yoruba sentences
eng_sents = [pair[0] for pair in data] # list of English sentences
yor_sents = [pair[1] for pair in data] # list of yoruba sentences
eng_vocab = build_vocab(eng_sents, 'eng') # id mapping for English
yor_vocab = build_vocab(yor_sents, 'yor') # id mapping for Yoruba

print(f"English Vocabulary Size: {len(eng_vocab)}")
print(f"Yoruba Vocabulary Size: {len(yor_vocab)}")

# Convert sentences to indices
def sentence_to_indices(sent, vocab):
    indices = [vocab.get(token, vocab.get("<UNK>", 0)) for token in sent.split()] # get returns the value of the key if exist.
    indices = [vocab["<SOS>"]] + indices + [vocab["<EOS>"]]
    return indices

# Prepare padded data
def prepare_data(data, eng_vocab, yor_vocab):
    src_data = [sentence_to_indices(pair[0], eng_vocab) for pair in data] # Converts English text → list of integers.
    tgt_data = [sentence_to_indices(pair[1], yor_vocab) for pair in data] # Converts Yoruba text → list of integers.
    # Pad sequences
    # pad_sequences() turns variable length sequences into equal length arrays; Because neural networks require fixed input size.
    
    src_padded = tf.keras.preprocessing.sequence.pad_sequences(src_data, padding='post', value = eng_vocab["<PAD>"])
    tgt_padded = tf.keras.preprocessing.sequence.pad_sequences(tgt_data, padding='post', value = yor_vocab["<PAD>"])
    return src_padded, tgt_padded

src_data, tgt_data = prepare_data(data, eng_vocab, yor_vocab)

print(f"Source data shape: {src_data.shape}")
print(f"Target data shape: {tgt_data.shape}")

# Step 2: Define the Encoder
class Encoder(tf.keras.Model):
    def __init__(self, input_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_size, embed_size) # Converts integers (word indices) → dense vectors.
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_state=True) # These states are the context the decoder will use.

    def call(self, x): # This defines how the encoder processes input during training & inference.
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_size] # Converts integer sequences → continuous vectors.
        _, hidden, cell = self.lstm(embedded)  # hidden, cell: [batch_size, hidden_size] # _ means "ignore the first output"
        return hidden, cell

# Step 3: Define the Decoder
class Decoder(tf.keras.Model):
    def __init__(self, output_size, embed_size, hidden_size):
        super(Decoder, self).__init__() # Initializes the internal Keras machinery
        self.embedding = tf.keras.layers.Embedding(output_size, embed_size) # Convert target-language input tokens (Yorùbá) → dense vectors.
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(output_size)

    def call(self, x, hidden, cell):
        embedded = self.embedding(x)  # [batch_size, 1, embed_size]
        lstm_out, hidden, cell = self.lstm(embedded, initial_state=[hidden, cell])
        output = self.fc(lstm_out)  # [batch_size, 1, output_size]
        return output, hidden, cell

# Step 4: Define the Seq2Seq Model
class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False, teacher_forcing_ratio=0.5):
        src, tgt = inputs
        batch_size = tf.shape(src)[0]
        tgt_len = tf.shape(tgt)[1]
        tgt_vocab_size = len(yor_vocab)

        # Encode the input
        hidden, cell = self.encoder(src)

        # prepare the output container
        outputs = []

        # Start with <SOS>
        input_token = tgt[:, 0:1]  # [batch_size, 1]

        # Decode  step
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs.append(output)

            if training:
                teacher_force = random.random() < teacher_forcing_ratio
                input_token = tgt[:, t:t+1] if teacher_force else tf.argmax(output, axis=-1, output_type=tf.int32)
            else:
                input_token = tf.argmax(output, axis=-1, output_type=tf.int32)

        # Concatenate all outputs
        if outputs:
            return tf.concat(outputs, axis=1)
        else:
            # Return zeros if no outputs
            return tf.zeros((batch_size, 1, tgt_vocab_size)) # This should never happen unless the sequence length = 1

# Step 5: Training
def train(model, src_data, tgt_data, epochs=100):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # If training is unstable, reduce to 0.001
    # Fix: Use the correct loss function for newer Keras versions
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def train_step(src, tgt):
        with tf.GradientTape() as tape:
            outputs = model([src, tgt], training=True, teacher_forcing_ratio=1.0)

            # Get target labels (excluding the first SOS token)
            target_labels = tgt[:, 1:]

            # Ensure outputs match the target sequence length
            seq_len = tf.shape(target_labels)[1]
            outputs = outputs[:, :seq_len, :]

            # Create mask to ignore padding tokens
            mask = tf.cast(target_labels != yor_vocab["<PAD>"], tf.float32)

            # Calculate loss
            loss = loss_fn(target_labels, outputs)
            loss = loss * mask

            # Calculate mean loss (only over non-padded tokens)
            total_loss = tf.reduce_sum(loss)
            total_tokens = tf.reduce_sum(mask)
            mean_loss = total_loss / (total_tokens + 1e-8)  # Add small epsilon to avoid division by zero

        gradients = tape.gradient(mean_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    print("Starting training...")
    for epoch in range(epochs):
        loss = train_step(src_data, tgt_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

    print("Training completed!")

# Initialize model
input_size = len(eng_vocab)
output_size = len(yor_vocab)
embed_size = 50
hidden_size = 100

print(f"\nInitializing model with:")
print(f"Input vocab size: {input_size}")
print(f"Output vocab size: {output_size}")
print(f"Embedding size: {embed_size}")
print(f"Hidden size: {hidden_size}")

encoder = Encoder(input_size, embed_size, hidden_size)
decoder = Decoder(output_size, embed_size, hidden_size)
model = Seq2Seq(encoder, decoder)

# Build the model by calling it once
dummy_src = tf.zeros((1, 5), dtype=tf.int32) # This creates a fake English input tensor just to "warm up" the model.
dummy_tgt = tf.zeros((1, 5), dtype=tf.int32) # they represent Yorùbá tokens.
_ = model([dummy_src, dummy_tgt], training=False)

print(f"Model built successfully!")
print(f"Total trainable parameters: {sum([tf.size(var).numpy() for var in model.trainable_variables])}")

# Train the model
train(model, src_data, tgt_data, epochs=50)

# Step 6: Inference
def translate(model, sentence, eng_vocab, yor_vocab, max_len=15):
    """Translate an English sentence to yoruba"""
    print(f"\nTranslating: '{sentence}'")

    # Tokenize and convert to indices
    tokens = sentence.split()
    indices = [eng_vocab["<SOS>"]] + [eng_vocab.get(token, 0) for token in tokens] + [eng_vocab["<EOS>"]]
    
    # Pad to match training data format
    src_seq = tf.keras.preprocessing.sequence.pad_sequences([indices], padding='post', value=eng_vocab["<PAD>"])
    
    # Create reverse vocabulary for Yoruba
    yor_idx_to_word = {v: k for k, v in yor_vocab.items()}
    
    # Encode
    hidden, cell = model.encoder(src_seq)
    
    # Start decoding with <SOS>
    input_token = tf.constant([[yor_vocab["<SOS>"]]])
    result = []
    
    for _ in range(max_len):
        output, hidden, cell = model.decoder(input_token, hidden, cell)
        predicted_id = tf.argmax(output, axis=-1).numpy()[0, 0]
        
        if predicted_id == yor_vocab["<EOS>"]:
            break
            
        if predicted_id != yor_vocab["<PAD>"]:
            word = yor_idx_to_word.get(predicted_id, "<UNK>")
            result.append(word)
        
        input_token = tf.constant([[predicted_id]])
    
    translation = " ".join(result)
    print(f"Translation: {translation}")
    return translation

# Test translations
test_sentences = ["I am happy", "You are sad", "We are ready"]
for sentence in test_sentences:
    translate(model, sentence, eng_vocab, yor_vocab)

print("\n" + "="*50)
print("HACKATHON PROJECT: English-Yoruba Neural Translation")
print("="*50)
print("Features:")
print("- Seq2Seq architecture with LSTM")
print("- Encoder-Decoder model")
print("- Teacher forcing during training")
print("- Custom vocabulary building")
print("- Padding for variable length sequences")
print("="*50)