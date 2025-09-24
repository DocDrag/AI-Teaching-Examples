import deepcut
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

MAXLEN = 50

# โหลด Word2Vec + โมเดล LSTM
w2v_model = Word2Vec.load("model/corpus.th.model")
embedding_dim = w2v_model.vector_size

# word2id
word_index = {word: i+1 for i, word in enumerate(w2v_model.wv.index_to_key)}

# โหลดโมเดล LSTM ที่เราเทรนไว้
model = load_model("model/final_lstm_model.h5")

# label map
id2label = {0: "positive", 1: "negative", 2: "neutral"}

def preprocess_text(text, maxlen=MAXLEN):
    # ตัดคำด้วย deepcut
    tokens = deepcut.tokenize(text)
    seq = [word_index.get(word, 0) for word in tokens]
    return pad_sequences([seq], maxlen=maxlen)

def predict(text):
    X = preprocess_text(text)
    probs = model.predict(X, verbose=0)[0]
    pred_id = np.argmax(probs)
    confidence = probs[pred_id]
    return id2label[pred_id], confidence

if __name__ == "__main__":
    while True:
        text = input("ป้อนประโยค (หรือ 'exit' เพื่อออก): ")
        if text.lower() == "exit":
            break
        label, conf = predict(text)
        print(f"👉 ผลลัพธ์: {label} (ความมั่นใจ {conf:.2f})\n")
