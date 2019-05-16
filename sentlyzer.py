from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import spacy, json
import string

# loading spaCy model during initialization
punctuations = string.punctuation
parser = spacy.load('en')

class sentilyze:

    def __init__(self):

        # provide training dataset path
        self.train ="/home/rahul/PycharmProjects/projects/train2.txt"


    def predict(self, text):
        """
        :param text: input text for sentiment Analysis
        :return: returns setiment of the text.
        """

        """Custom transformer using spaCy"""
        class predictors(TransformerMixin):
            def transform(self, X, **transform_params):
                return [clean_text(text) for text in X]

            def fit(self, X, y=None, **fit_params):
                return self

            def get_params(self, deep=True):
                return {}

        # Basic utility function to clean the text
        def clean_text(text):
            return text.strip().lower()

        # Create spacy tokenizer that parses a sentence and generates tokens
        # these can also be replaced by word vectors
        def spacy_tokenizer(sentence):
            tokens = parser(sentence)
            tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
            tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
            return tokens

        # create vectorizer object to generate feature vectors, we will use custom spacy’s tokenizer
        vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
        classifier = LinearSVC()

        # Create the  pipeline to clean, tokenize, vectorize, and classify
        pipe = Pipeline([("cleaner", predictors()),
                         ('vectorizer', vectorizer),
                         ('classifier', classifier)])

        ## loading training label dataset here
        with open(self.train, 'r') as file:
            data = json.load(file)

        train = data["train"]
        test = [(text, ' ')]

        # Create model and measure accuracy
        model = pipe.fit([x[0] for x in train], [x[1] for x in train])

        pred_data = model.predict([x[0] for x in test])
        for (sample, pred) in zip(test, pred_data):

            return pred
