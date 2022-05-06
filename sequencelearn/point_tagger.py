from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sequencelearn import CONSTANT_OUTSIDE, PointTagger

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


class SupportVectorTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = SVC(**kwargs)


class LogisticTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = LogisticRegression(**kwargs)


class NearestNeighborTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = KNeighborsClassifier(**kwargs)


class BayesTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = GaussianNB(**kwargs)


class GaussianTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = GaussianProcessClassifier(**kwargs)


class TreeTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = DecisionTreeClassifier(**kwargs)


class ForestTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = RandomForestClassifier(**kwargs)


class AdaTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = AdaBoostClassifier(**kwargs)
