import numpy

from ._instance_grouping_model import _InstanceGroupingModel
from ._regression_model_table import _RegressionModelTable


class EnsembleModel(object):

    def __init__(self, cluster_model, regression_models, classifier_model=None, adapt_data_for_cluster_function=None,
                 function_to_evaluate_model=None, function_to_select_model=None):
        # cluster_model: object with .fit(X) and .predict(X) methods
        # regression_models: object or list of objects with .fit(X, y) and .predict(X, n_samples)
        # classifier_model: None or object with .fit(X, y) and .predict(X) methods. If None, cluster.predict is applied
        # adapt_data_for_cluster_function: function(X), returns adapted X (it could reduce variable dimensions)
        # function_to_evaluate_model: None or function(model, X, y), returns the model_evaluation of the model object
        # function_to_select_model: None or function([model]], [model_evaluation]), returns selected model

        self._grouping_model = _InstanceGroupingModel(cluster_model, classifier_model=classifier_model,
                                                      adapt_data_for_cluster_function=adapt_data_for_cluster_function)

        self._regression_model_table = _RegressionModelTable(regression_models, function_to_evaluate_model,
                                                             function_to_select_model=function_to_select_model)

        self._flag_evaluate_regression_models = function_to_evaluate_model is not None
        self._flag_fit = False
        self._flag_auto_label = True
        self._label_values = None
        self._num_vars = None
        self._instance_window_size = None

    @property
    def cluster_model(self):
        return self._grouping_model.cluster_model

    @property
    def classifier_model(self):
        return self._grouping_model.classifier_model

    @property
    def num_vars(self):
        return self._num_vars

    @property
    def instance_window_size(self):
        return self._instance_window_size

    @property
    def model_regression_table(self):
        return self._regression_model_table.pandas_table

    def fit(self, x, y, extra_var_matrix=None, instance_labels=None):
        self._flag_fit = True
        self._set_model_params(x, extra_var_matrix, instance_labels)
        if self._flag_auto_label:
            instance_labels = self._grouping_model.fit(x, extra_var_matrix=extra_var_matrix)
            self._label_values = numpy.unique(instance_labels).tolist()
        self._fit_regression_models(x, y, instance_labels)
        return self

    def _set_model_params(self, x, extra_var_matrix=None, instance_labels=None):
        self._instance_window_size = x.shape[1]
        self._num_vars = x.shape[2]
        if instance_labels is not None:
            self._flag_auto_label = False
        if extra_var_matrix is not None:
            if extra_var_matrix.ndim != 2:
                raise ValueError("extra_var_matrix must have two dimensions (n_instances, n_extra_features)")
            self._extra_var_dim = extra_var_matrix.shape[1]

    def _fit_regression_models(self, x, y, instance_labels):
        n_vars = x.shape[2]
        self._initialize_regression_models_table(x.shape[2])
        labeled_data_tuples = [self._split_cluster(x, y, instance_labels, LABEL) for LABEL in self._label_values]
        for IDX, LABEL_VALUE in enumerate(self._label_values):
            x_label, y_label = labeled_data_tuples[IDX]
            for VAR_IDX in range(n_vars):
                label_name = self._return_label_name(LABEL_VALUE, VAR_IDX)
                self._fit_regression_model(x_label[:, :, VAR_IDX], y_label[:, :, VAR_IDX], label_name)

    def _initialize_regression_models_table(self, n_vars):
        label_names = [self._return_label_name(LABEL, VAR) for LABEL in self._label_values for VAR in range(n_vars)]
        self._regression_model_table.initialize_tables(label_names)

    def _fit_regression_model(self, x, y, label_name):
        print("Training cluster label: %s, num_instances: %s" % (label_name, x.shape[0]))
        if self._flag_evaluate_regression_models:
            self._regression_model_table.evaluate_label_models(x, y, label_name)
        reg_model = self._regression_model_table.return_selected_label_model(label_name)
        fit_reg_model = reg_model.fit(x, y)
        self._regression_model_table.set_label_regression_model(fit_reg_model, label_name)

    def predict(self, x, n_samples, extra_var_matrix=None, instance_labels=None):
        if instance_labels is None:
            if not self._flag_auto_label:
                raise ValueError("Cluster and classifier were not fit. Instance labels must be introduced.")
            instance_labels = self.classify_instances(x, extra_var_matrix=extra_var_matrix)
        y = self._apply_regression_models(x, instance_labels, n_samples)
        return y  # Matriz (X.shape[0], H)

    def classify_instances(self, x, extra_var_matrix=None):
        return self._grouping_model.predict(x, extra_var_matrix=extra_var_matrix)

    @staticmethod
    def _split_cluster(x, y, instance_labels, label):
        logical_instances = instance_labels == label
        return x[logical_instances, :, :], y[logical_instances, :, :]

    def _apply_regression_models(self, x, instance_labels, n_samples):
        n_instances = x.shape[0]
        n_vars = x.shape[2]
        output = numpy.zeros((n_instances, n_samples, n_vars))
        for LABEL_VALUE in self._label_values:
            log_instances = instance_labels == LABEL_VALUE
            if not log_instances.any():
                continue
            for VAR_IDX in range(n_vars):
                label_name = self._return_label_name(LABEL_VALUE, VAR_IDX)
                output[log_instances, :, VAR_IDX] = self._apply_regression_model(x[log_instances, :, VAR_IDX], n_samples, label_name)
        return output

    def _apply_regression_model(self, x, n_samples, label_name):
        reg_model = self._regression_model_table.return_label_regression_model(label_name)
        return numpy.apply_along_axis(lambda m: reg_model.predict(m, n_samples), 1, x)

    @staticmethod
    def _return_label_name(label_value, num_var):
        return "L" + str(label_value) + "_V" + str(num_var)
