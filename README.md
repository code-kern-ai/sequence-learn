![sequence-learn](https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/6274762101c203108c785958_banner.png)
<img src="https://img.shields.io/badge/version-0.0.5-green" />
<img src="https://img.shields.io/badge/pip-0.0.5-green" />

# ➡️ sequence-learn
With `sequence-learn`, you can build models for named entity recognition as quickly as if you were building a sklearn classifier.

It takes as input embedded token lists, which you can create within a few lines of code using the [embedders library](https://github.com/code-kern-ai/embedders). The labels are on token-level, i.e., for each token, you must provide some information in a simple list.

## Installation
You can set up this library via either running `$ pip install sequencelearn`, or via cloning this repository and running `$ pip install -r requirements.txt` in your repository.

A sample installation including `embedders` would be (including [spaCy](https://github.com/explosion/spaCy) for tokenization):
```
$ conda create --name sequence-learn python=3.9
$ conda activate sequence-learn
$ pip install sequencelearn
$ pip install embedders
$ python -m spacy download en_core_web_sm
```

## Usage
Once you have installed the package(s), you can easily create the input for a text corpus and put it - together with the required labels - into the model training.

```python
from embedders.extraction.contextual import TransformerTokenEmbedder
from sequencelearn.sequence_tagger import CRFTagger

corpus = [
    "I went to Cologne in 2009",
    "My favorite number is 41",
    # ...
]

labels = [
    ["OUTSIDE", "OUTSIDE", "OUTSIDE", "CITY", "OUTSIDE", "YEAR"],
    ["OUTSIDE", "OUTSIDE", "OUTSIDE", "OUTSIDE", "DIGIT"],
    # ...
]

# use embedders to easily convert your raw data
embedder = TransformerTokenEmbedder("distilbert-base-uncased", "en_core_web_sm")

embeddings = embedder.encode(corpus)
# contains a list of ragged shape [num_texts, num_tokens (text-specific), embedding_dimension]

tagger = CRFTagger()
tagger.fit(embeddings, labels)
```

Now that you've trained a tagger model, you can easily apply it to new text data.

```python
sentence = ["My birthyear is 2002"]
print(tagger.predict(embedder.encode(sentence, fit_model=False)))
# prints [['OUTSIDE', 'OUTSIDE', 'OUTSIDE', 'YEAR']]
```

## Roadmap
- [ ] Add extensive documentation to existing models
- [ ] Add sample projects
- [x] Add sequence-based models (e.g. RNNs)
- [ ] Add label conversions for different formats
- [ ] Add further interface capabilities

If you want to have something added, feel free to open an [issue](https://github.com/code-kern-ai/sequence-learn/issues).

## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

And please don't forget to leave a ⭐ if you like the work! 

## License
Distributed under the Apache 2.0 License. See LICENSE.txt for more information.

## Contact
This library is developed and maintained by [kern.ai](https://github.com/code-kern-ai). If you want to provide us with feedback or have some questions, don't hesitate to contact us. We're super happy to help ✌️

## Acknowledgements
Huge thanks to [Erik Ziegler](https://github.com/erksch) for helping with the CRF implementation!
