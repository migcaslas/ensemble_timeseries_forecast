import numpy
from .adapt_grouping_matrix import AdaptGroupingMatrix as AdaptMat
from ._check_predefined_models import _CheckPredefinedModels


class _InstanceGroupingModel(object):
    def __init__(self, cluster_model, classifier_model=None, adapt_data_for_cluster_function=None):
        # if cluster_model is None or not _CheckPredefinedModels.check_valid_model(cluster_model):
        #     raise ValueError("cluster_model is not valid. It must have .fit and .predict methods.")
        if classifier_model is not None and not _CheckPredefinedModels.check_valid_model(classifier_model):
            raise ValueError("classifier_model is not valid. It must have .fit and .predict methods.")
        # Los métodos de clustering y clasificación actuan sobre una matriz (n_instances, n_features)
        # Por defecto la matriz es (n_instances, n_variables * window_size)
        self._cluster_model = cluster_model
        self._classifier_model = classifier_model

        # Los datos de entrada para la funcion que adapta los datos es (n_instances, window_size, n_variables)
        self._adapt_group_data_function = adapt_data_for_cluster_function
        # Si se usa una extra var para el fit deberán utilizarse las mismas para el predict
        self._extra_var_dim = None

    @property
    def cluster_model(self):
        return self._cluster_model

    @property
    def classifier_model(self):
        return self._classifier_model

    def fit(self, x, extra_var_matrix=None):
        x_group = AdaptMat.adapt_windowed_data(x, self._adapt_group_data_function)
        x_group = AdaptMat.return_classification_matrix(x_group, self._extra_var_dim, extra_var_matrix)
        instance_labels = self._cluster_instances(x_group)
        if self._classifier_model is not None:
            self._train_classifier(x_group, instance_labels)
        return instance_labels

    def predict(self, x, extra_var_matrix=None):
        x_group = AdaptMat.adapt_windowed_data(x, self._adapt_group_data_function)
        x_group = AdaptMat.return_classification_matrix(x_group, self._extra_var_dim, extra_var_matrix)
        instance_labels = self._classify_instances(x_group)
        return instance_labels

    def _cluster_instances(self, x_group):
        # if not _CheckPredefinedModels.check_valid_model(self._cluster_model):
        #     raise ValueError("Impossible to fit the cluster, wrong methods.")
        self._cluster_model = self._cluster_model.fit(x_group)
        labels = self._cluster_model.labels_
        self._label_values = numpy.unique(labels).tolist()
        return labels

    def _train_classifier(self, x_group, y):
        if self._classifier_model is None or not _CheckPredefinedModels.check_valid_model(self._classifier_model):
            raise ValueError("Impossible to train None classifier")
        self._classifier_model = self._classifier_model.fit(x_group, y)

    def _classify_instances(self, x_group):
        if self._classifier_model is not None:
            return self._classifier_model.predict(x_group)
        else:
            return self._cluster_model.predict(x_group)
