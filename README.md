# sequence-learn
Sklearn-like API for Sequence Learning tasks like Named Entity Recognition.

`sequence-learn` takes as input embedded token lists, which you can produce using e.g. Spacy or NLTK for tokenization and Sklearn or Hugging Face for the embedding procedure. The labels are on token-level, i.e., for each token, you must provide some information in a simple list.

## How to install
You can set up this library via either running `pip install sequencelearn`, or via cloning this repository and running `pip install -r requirements.txt` in your repository.

This works great together with the [embedders](https://github.com/code-kern-ai/embedders) library, which converts your documents into embeddings within only a few lines of code.

**Caution:** We currently have this tested for Python 3 up to Python 3.9. If your installation runs into issues, please contact us.

## Example
```python
from embedders.extraction.count_based import CharacterTokenEmbedder
from sequencelearn.point_tagger import TreeTagger

corpus = [
    "I went to Cologne in 2009",
    "My favorite number is 41",
]

labels = [
    ["OUTSIDE", "OUTSIDE", "OUTSIDE", "CITY", "OUTSIDE", "YEAR"],
    ["OUTSIDE", "OUTSIDE", "OUTSIDE", "OUTSIDE", "DIGIT"]
]

embedder = CharacterTokenEmbedder("en_core_web_sm")
embeddings = embedder.encode(corpus) # contains a list of ragged shape [num_texts, num_tokens (text-specific), embedding_dimension]

tagger = TreeTagger()
tagger.fit(embeddings, labels)

sentence = "My birthyear is 1998"
print(tagger.predict([sentence]))
```

## How to contribute
Currently, the best way to contribute is via adding issues for the kind of transformations you like and starring this repository :-)