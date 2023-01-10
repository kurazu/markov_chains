# Simple Markov Chain text generation

## Dataset

To train the model a dataset of [Sherlock Holmes stories](https://www.kaggle.com/datasets/idevji1/sherlock-holmes-stories?resource=download) can be used.

## Usage

```bash
poetry run python chains.py --input-directory=./sherlock --n-gram=3 sherlock holmes was
```

## Output

```text
sherlock holmes was on a visit , and will be away all the evening , and he had jumped off from the road and telephone to make sure that all was right with
sherlock holmes was a man who made his mark quickly . wherever he was the keeper of a low , clear whistle . i am much indebted to you . " " what
sherlock holmes was as good as his word , and 1914 for the figures . " try the settee , " said our prisoner , i think - - and we waited for
sherlock holmes was that , by taking train , we might talk it over while we drive . " a white cock , " said lestrade . " oh , a few of
sherlock holmes was transformed when he was searched . six plaster casts of the famous official was a white - faced , elderly gentleman with fiery red hair . with an apology he
```