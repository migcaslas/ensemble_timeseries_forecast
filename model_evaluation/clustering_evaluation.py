import pandas
import numpy

from window_functions.windowing import Windowing


class ClusteringEvaluation(object):

    @classmethod
    def classify_labels_time_series(cls, time_series, model, step):
        instance_labels = cls._forecast_selected_time_series(time_series, model, step)
        return pandas.Series(instance_labels, index=time_series.index[:len(instance_labels)], name="labels")

    @classmethod
    def _forecast_selected_time_series(cls, time_series, model, step):
        data = time_series.to_numpy()
        windowing_function = Windowing.function_windowing(model.instance_window_size, step)
        instances = windowing_function(data)
        label_instances = model.classify_instances(instances)  # The last one is not necessary
        extra_ = numpy.repeat(numpy.nan, model.instance_window_size-step)
        repeat_instances = numpy.tile(label_instances, (step, 1)).transpose().reshape(-1)
        return numpy.concatenate((extra_, repeat_instances))

