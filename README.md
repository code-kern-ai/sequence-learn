# sequence-learn
Sklearn-like API for Sequence Learning tasks like Named Entity Recognition.

`sequence-learn` takes as input embedded token lists, which you can produce using e.g. Spacy or NLTK for tokenization and Sklearn or Hugging Face for the embedding procedure. The labels are on token-level, i.e., for each token, you must provide some information in a simple list.

## Example
```python
# some token-level embedding, e.g. based on character embeddings
x = [[
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
],[
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0]
]]

# token-level labels, where OUTSIDE means that this token contains no label
y = [["OUTSIDE", "LABEL-1"],
     ["LABEL-2","LABEL-1","OUTSIDE"]]
```
