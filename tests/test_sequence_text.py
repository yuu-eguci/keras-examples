import keras
import numpy as np
import pytest

from conftest import load_example

BABI_TRAIN = """1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary? \tbathroom\t1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel? \thallway\t4
1 John moved to the kitchen.
2 Where is John? \tkitchen\t1
"""
BABI_TEST = """1 Sandra went to the kitchen.
2 Where is Sandra? \tkitchen\t1
"""


@pytest.mark.parametrize("script, epochs", [
    ("lstm/alphabet_lstm.py", 300),
    ("lstm/lstm_qa.py", 40),
    ("nlp/reuters.py", 100),
    ("word2vec/word_embedding.py", 5000),
])
def test_default_epochs(script, epochs):
    assert load_example(script).parse_args([]).epochs == epochs


def test_alphabet_dataset_pairs_each_letter_with_the_next():
    x, y = load_example("lstm/alphabet_lstm.py").make_dataset()

    assert x.shape == (25, 1, 1)
    np.testing.assert_allclose(x[:, 0, 0], np.arange(25) / 26)
    np.testing.assert_array_equal(y, np.arange(1, 26))


def test_alphabet_model_is_stateful_with_batch_size_one():
    model = load_example("lstm/alphabet_lstm.py").build_model()

    lstm = model.get_layer("lstm")
    assert lstm.stateful
    assert model.input_shape == (1, 1, 1)
    assert model.output_shape == (1, 26)


def test_alphabet_main_prints_predicted_sequences(capsys):
    load_example("lstm/alphabet_lstm.py").main(["--epochs", "1"])

    out = capsys.readouterr().out
    assert "Model accuracy (train data):" in out
    assert out.count(" -> ") == 25 + 5
    assert "New start: K" in out


def test_tokenize_splits_words_and_punctuation():
    tokens = load_example("lstm/lstm_qa.py").tokenize("Bob dropped the apple. Where is the apple?")

    assert tokens == ["Bob", "dropped", "the", "apple", ".", "Where", "is", "the", "apple", "?"]


def test_parse_stories_collects_story_before_each_question():
    module = load_example("lstm/lstm_qa.py")

    data = module.parse_stories(BABI_TRAIN.splitlines())

    assert len(data) == 3
    story, question, answer = data[1]
    assert story == [["Mary", "moved", "to", "the", "bathroom", "."],
                     ["John", "went", "to", "the", "hallway", "."],
                     ["Daniel", "went", "back", "to", "the", "hallway", "."],
                     ["Sandra", "moved", "to", "the", "garden", "."]]
    assert question == ["Where", "is", "Daniel", "?"]
    assert answer == "hallway"


@pytest.fixture
def babi_dir(tmp_path):
    task_dir = tmp_path / "en-10k"
    task_dir.mkdir()
    (task_dir / "qa1_single-supporting-fact_train.txt").write_text(BABI_TRAIN)
    (task_dir / "qa1_single-supporting-fact_test.txt").write_text(BABI_TEST)
    return tmp_path


def test_lstm_qa_sums_story_and_question_encodings():
    model = load_example("lstm/lstm_qa.py").build_model(vocab_size=20, story_maxlen=12, query_maxlen=4)

    assert [tuple(shape) for shape in model.input_shape] == [(None, 12), (None, 4)]
    assert model.output_shape == (None, 20)
    assert any(isinstance(layer, keras.layers.Add) for layer in model.layers)


def test_lstm_qa_main_trains_and_reports_test_accuracy(babi_dir, capsys):
    load_example("lstm/lstm_qa.py").main(["--data-dir", str(babi_dir), "--epochs", "1"])

    assert "Test accuracy:" in capsys.readouterr().out


def test_reuters_vectorizes_word_presence():
    module = load_example("nlp/reuters.py")

    matrix = module.to_binary_matrix([[1, 3, 3], [2]], 4)

    np.testing.assert_array_equal(matrix, [[0, 1, 0, 1], [0, 0, 1, 0]])


def test_reuters_main_uses_top_words_and_reports_accuracy(monkeypatch, capsys):
    calls = []
    rng = np.random.default_rng(0)

    def load_data(**kwargs):
        calls.append(kwargs)
        sequences = [list(rng.integers(1, 1000, size=rng.integers(3, 20))) for _ in range(40)]
        labels = rng.integers(0, 46, size=40)
        return (sequences[:30], labels[:30]), (sequences[30:], labels[30:])

    monkeypatch.setattr(keras.datasets.reuters, "load_data", load_data)

    load_example("nlp/reuters.py").main(["--epochs", "1"])

    assert calls == [{"num_words": 1000, "test_split": 0.2}]
    assert "Test acc:" in capsys.readouterr().out


def test_word_embedding_prints_vector_for_every_word(capsys):
    module = load_example("word2vec/word_embedding.py")

    module.main(["--epochs", "1"])

    lines = [line for line in capsys.readouterr().out.splitlines() if ": [" in line]
    words = sorted({word for sentence in module.SENTENCES for word in sentence.split()})
    assert [line.split(":")[0] for line in lines] == words
    for line in lines:
        values = line.split("[")[1].rstrip("]").split()
        assert len(values) == 3
        [float(value) for value in values]
