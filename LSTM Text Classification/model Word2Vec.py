import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    inp = "word.txt"
    output_dir = "model"
    output_file = os.path.join(output_dir, "corpus.th.model")

    program = os.path.basename(sys.argv[0])
    model = Word2Vec(
        LineSentence(inp),
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,
        workers=multiprocessing.cpu_count()
    )

    # ตรวจสอบว่าไดเรกทอรี 'model' มีอยู่หรือไม่ ถ้าไม่มี ให้สร้างขึ้นมา
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    model.save(output_file)
    model.wv.save_word2vec_format(os.path.join(output_dir, "corpus.th.vec"), binary=False)