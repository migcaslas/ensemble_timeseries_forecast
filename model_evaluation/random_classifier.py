import numpy


class RandomClassifier(object):
    def __init__(self):
        self.group_values = None

    def fit(self, x_group, y):
        self.group_values = numpy.unique(y)
        return self

    def predict(self, x_group):
        values = numpy.apply_along_axis(lambda x: numpy.random.choice(self.group_values), arr=x_group, axis=1)
        return values.reshape(-1)
