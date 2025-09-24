import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec

# โหลด Word2Vec
w2v_model = Word2Vec.load("model/corpus.th.model")
embedding_dim = w2v_model.vector_size

# word2id
word_index = {word: i+1 for i, word in enumerate(w2v_model.wv.index_to_key)}
vocab_size = len(word_index) + 1

# matrix embedding
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# โหลด dataset (tokenized แล้ว)
texts, labels = [], []
with open("dataset_format.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if "<LABEL>" not in line:
            continue
        text, label = line.split("<LABEL>")
        texts.append(text.strip())
        labels.append(label.strip())

# แปลง label เป็นตัวเลข
label2id = {"positive":0, "negative":1, "neutral":2}
y = [label2id[l] for l in labels]

sequences = []
for text in texts:
    seq = [word_index.get(word, 0) for word in text.split()]
    sequences.append(seq)

# padding (กำหนด maxlen=50)
maxlen = 50
X = pad_sequences(sequences, maxlen=maxlen)
y = to_categorical(y, num_classes=len(label2id))

# สร้าง BiLSTM model
model = Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=maxlen,
    trainable=True
))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(len(label2id), activation="softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Callback สำหรับหยุดอัตโนมัติ + เซฟโมเดล
os.makedirs("model", exist_ok=True)

es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
ckpt = ModelCheckpoint("model/best_lstm_model.h5",
                       save_best_only=True,
                       monitor="val_loss")

# เทรน
history = model.fit(
    X, y,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es, ckpt]
)

# บันทึกโมเดลสุดท้าย
model.save("model/final_lstm_model.h5")
