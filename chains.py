import collections
import logging
import operator
import random
from pathlib import Path
from typing import Callable, Iterable

import click
import more_itertools as mit
import tensorflow as tf
import tensorflow_text as text
from returns.curry import partial
from returns.pipeline import pipe

logger = logging.getLogger(__name__)
WordType = tuple[int, ...]


@click.command()
@click.option(
    "--input-directory",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path
    ),
    required=True,
    default="sherlock",
)
@click.option(
    "--tokenizer-vocab",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
    required=True,
    default="vocab.txt",
)
@click.option("--n-gram", type=int, required=True, default=2)
@click.option("--words", type=int, required=True, default=30)
@click.option("--sentences", type=int, required=True, default=5)
def main(
    input_directory: Path,
    tokenizer_vocab: Path,
    n_gram: int,
    words: int,
    sentences: int,
) -> None:
    tokenizer = text.BertTokenizer(str(tokenizer_vocab), lower_case=True)
    to_tokens: Callable[[str], Iterable[WordType]] = pipe(  # type: ignore
        tokenizer.tokenize,
        operator.methodcaller("to_list"),
        operator.itemgetter(0),
        partial(map, tuple),
    )
    counts: dict[tuple[WordType, ...], dict[WordType, int]] = collections.defaultdict(
        partial(collections.defaultdict, int)
    )
    logger.debug("Building counts")
    for file in input_directory.glob("*.txt"):
        logger.debug("Processing file %s", file)
        story = file.read_text()
        story = story[: story.find("----------")]
        tokens = to_tokens(story)
        for window in mit.sliding_window(tokens, n_gram + 1):
            *prev_words, next_word = window
            counts[tuple(prev_words)][next_word] += 1
    probabilities: dict[tuple[WordType, ...], tuple[list[WordType], list[int]]] = {
        prev_words: (list(next_mapping.keys()), list(next_mapping.values()))
        for prev_words, next_mapping in counts.items()
    }
    logger.debug("Ready for input")
    while True:
        seed_text = click.prompt(f">>> Seed (starting fragment with {n_gram} words)")
        seed_tokens = tuple(to_tokens(seed_text))
        if len(seed_tokens) != n_gram:
            click.echo(f"Seed must have {n_gram} words")
            continue

        if seed_tokens not in probabilities:
            click.echo(
                "Seed unrecognized, try a different phrase "
                "that can be found in the corpus"
            )
            continue

        for _ in range(sentences):
            history = collections.deque(seed_tokens, maxlen=n_gram)
            tokens = list(history)
            for _ in range(words):
                seed = tuple(history)
                try:
                    values, probs = probabilities[seed]
                except KeyError:
                    break  # this is the end
                (next_word,) = random.choices(values, probs)
                tokens.append(next_word)
                history.append(next_word)
            tokens_tensor = tf.ragged.constant(tokens, dtype=tf.int64)
            word_tensor = tokenizer.detokenize(tokens_tensor)
            sentence_tensor = tf.strings.join(word_tensor, " ")
            sentence_bytes: bytes
            (sentence_bytes,) = sentence_tensor.numpy().tolist()
            sentence = sentence_bytes.decode("utf-8")
            click.echo(sentence)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s][%(levelname)8s][%(name)s] %(message)s",
    )
    logger.setLevel(logging.DEBUG)
    main()
