from sequencelearn.base import CONSTANT_OUTSIDE, PointTagger
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class TreeTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = DecisionTreeClassifier(**kwargs)


class ForestTagger(PointTagger):
    def __init__(self, constant_outside=CONSTANT_OUTSIDE, **kwargs):
        super().__init__(constant_outside)
        self.model = RandomForestClassifier(**kwargs)
