import tensorflow as tf
import random
import numpy as np

# Step 1: Data Preparation
data = [
    ("I am happy", "Mo dun"),
    ("You are sad", "O ni ibanuje"),
    ("She is tired", "O re"),
    ("We are learning", "A n ko eko"),
    ("They are working", "Won n sise")
]

print(f"Training data: {len(data)} sentence pairs")
for eng, yor in data:
    print(f"  '{eng}' -> '{yor}'")

# Build vocabularies
def build_vocab(sentences, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

eng_sentences = [pair[0] for pair in data]
yor_sentences = [pair[1] for pair in data]

eng_vocab = build_vocab(eng_sentences)
yor_vocab = build_vocab(yor_sentences)

print(f"\nEnglish vocabulary ({len(eng_vocab)} words): {eng_vocab}")
print(f"Yoruba vocabulary ({len(yor_vocab)} words): {yor_vocab}")

# Convert sentences to indices
def sentence_to_indices(sentence, vocab):
    return [vocab["<SOS>"]] + [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()] + [vocab["<EOS>"]]

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
    print(f"Input tokens: {tokens}")
    print(f"Input indices: {indices}")

    # Convert to tensor and add batch dimension
    src_tensor = tf.convert_to_tensor([indices], dtype=tf.int32)

    # Encode
    hidden, cell = model.encoder(src_tensor)
    print(f"Encoded to context vector of shape: {hidden.shape}")

    # Decode step by step
    input_token = tf.convert_to_tensor([[yor_vocab["<SOS>"]]], dtype=tf.int32)
    output_tokens = []

    print("Decoding steps:")
    for step in range(max_len):
        output, hidden, cell = model.decoder(input_token, hidden, cell)
        predicted_token = tf.argmax(output, axis=-1).numpy()[0, 0]

        # Get the word for this token
        inv_yor_vocab = {v: k for k, v in yor_vocab.items()}
        predicted_word = inv_yor_vocab.get(predicted_token, "<UNK>")

        print(f"  Step {step+1}: {predicted_word} (token {predicted_token})")

        if predicted_token == yor_vocab["<EOS>"]:
            print("  Reached EOS token, stopping")
            break

        output_tokens.append(predicted_token)
        input_token = tf.convert_to_tensor([[predicted_token]], dtype=tf.int32)

    # Convert indices to words
    inv_yor_vocab = {v: k for k, v in yor_vocab.items()}
    translated_words = [inv_yor_vocab.get(idx, "<UNK>") for idx in output_tokens]
    translation = " ".join(translated_words)

    print(f"Final translation: '{translation}'")
    return translation

# Test translations
print("\n" + "="*50)
print("TESTING TRANSLATIONS")
print("="*50)

# Test on training data
test_sentences = ["I am happy", "You are sad", "She is tired"]
for sentence in test_sentences:
    # Find expected translation
    expected = None
    for eng, yor in data:
        if eng == sentence:
            expected = yor
            break

    translation = translate(model, sentence, eng_vocab, yor_vocab)
    print(f"Expected: '{expected}'")
    print(f"Got:      '{translation}'")
    print("-" * 30)