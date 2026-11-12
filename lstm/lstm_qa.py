"""bAbI の質問応答タスクで、ストーリーと質問をそれぞれ LSTM で符号化し、答えの単語を予測する。

参考:
- https://www.youtube.com/watch?v=YimQOpSRULY&t=885s
- https://github.com/keras-team/keras/blob/2.3.1/examples/babi_rnn.py
"""
import argparse
import itertools
import os
import re

import keras
import numpy as np

EMBED_DIM = 64
BATCH_SIZE = 32


def tokenize(sentence):
    """文を単語と記号に分割する。

    >>> tokenize("Bob dropped the apple. Where is the apple?")
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split(r"(\W+)", sentence) if x.strip()]


def parse_stories(lines):
    """bAbI 形式の行から (ストーリーの文のリスト, 質問, 答え) のリストを作る。"""
    data = []
    story = []
    for line in lines:
        nid, line = line.strip().split(" ", 1)
        if int(nid) == 1:
            story = []
        if "\t" in line:
            question, answer, _ = line.split("\t")
            data.append(([sentence for sentence in story if sentence], tokenize(question), answer))
            # 質問の行も番号を数えるので、空の文を入れて番号をそろえる
            story.append([])
        else:
            story.append(tokenize(line))
    return data


def get_stories(filename):
    with open(filename, encoding="utf-8") as fp:
        data = parse_stories(fp.readlines())
    # ストーリーの文をつなげて1つの単語列にする
    return [(list(itertools.chain.from_iterable(story)), question, answer)
            for story, question, answer in data]


def vectorize_stories(data, word_index, story_maxlen, query_maxlen):
    stories = [[word_index[w] for w in story] for story, _, _ in data]
    queries = [[word_index[w] for w in query] for _, query, _ in data]
    answers = np.array([word_index[answer] for _, _, answer in data])
    return (keras.utils.pad_sequences(stories, maxlen=story_maxlen),
            keras.utils.pad_sequences(queries, maxlen=query_maxlen),
            answers)


def build_model(vocab_size, story_maxlen, query_maxlen):
    """ストーリーと質問をそれぞれ埋め込み・LSTM で符号化し、足し合わせて答えの単語を予測する。"""
    layers = keras.layers
    story = keras.Input(shape=(story_maxlen,))
    query = keras.Input(shape=(query_maxlen,))
    encoded_story = layers.LSTM(EMBED_DIM)(layers.Embedding(vocab_size, EMBED_DIM)(story))
    encoded_query = layers.LSTM(EMBED_DIM)(layers.Embedding(vocab_size, EMBED_DIM)(query))
    merged = layers.Add()([encoded_story, encoded_query])
    answer = layers.Dense(vocab_size, activation="softmax")(merged)
    return keras.Model([story, query], answer)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="tasks_1-20_v1-2")
    parser.add_argument("--task", default="qa1_single-supporting-fact")
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = os.path.join(args.data_dir, "en-10k", args.task + "_{}.txt")
    train_stories = get_stories(path.format("train"))
    test_stories = get_stories(path.format("test"))
    print("# of train stories:", len(train_stories))
    print("# of test stories:", len(test_stories))

    all_stories = train_stories + test_stories
    vocab = sorted({w for story, query, answer in all_stories for w in story + query + [answer]})
    # 番号0はパディング用に空けておく
    word_index = {word: i + 1 for i, word in enumerate(vocab)}
    vocab_size = len(vocab) + 1
    story_maxlen = max(len(story) for story, _, _ in all_stories)
    query_maxlen = max(len(query) for _, query, _ in all_stories)
    print("vocab size:", vocab_size, "story maxlen:", story_maxlen, "query maxlen:", query_maxlen)

    x_train, xq_train, y_train = vectorize_stories(train_stories, word_index, story_maxlen, query_maxlen)
    x_test, xq_test, y_test = vectorize_stories(test_stories, word_index, story_maxlen, query_maxlen)

    model = build_model(vocab_size, story_maxlen, query_maxlen)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit([x_train, xq_train], y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
              validation_split=0.05)

    loss, accuracy = model.evaluate([x_test, xq_test], y_test, batch_size=BATCH_SIZE, verbose=0)
    print("Test loss: %.4f / Test accuracy: %.4f" % (loss, accuracy))


if __name__ == "__main__":
    main()
