import numpy
from statsmodels.tsa.arima.model import ARIMA


class ArimaModel(object):
    def __init__(self, order_params, model_selection_function=None):
        self._order_params = order_params
        self._model = None
        # A esta funcion le pasas (X_train, y_train, exponential_params)
        self._model_selection_function = model_selection_function  # Puede ser None?
        self._instance_window_size = None

    @property
    def instance_window_size(self):
        return self._instance_window_size

    def __str__(self):
        return "ArimaModel(%s)" % self.params.__str__()

    @property
    def params(self):
        return self._order_params

    @classmethod
    def _return_model_function(cls, order_params):
        def fun(instance, y_values=None):
            return ARIMA(instance, order=order_params)
        return fun

    def _is_fit(self):
        return self._model is not None

    def fit(self, x_train, y_train):
        if self._is_fit():
            raise ValueError("Fit was already performed in this model.")
        new_model = self.__class__(self._order_params, model_selection_function=self._model_selection_function)
        new_model._fit(x_train, y_train)
        return new_model

    def _fit(self, x_train, y_train):
        if x_train.shape[0] < 1:
            raise ValueError("x_train must be a matrix with some instances.")
        if x_train.ndim == 1:
            get_model_func = self._return_model_function(self._order_params)
            arima_model = get_model_func(x_train)
            self._model = arima_model.fit()
        else:
            self._instance_window_size = x_train.shape[1]
            func_to_get_model = self._func_to_generate_model(self._order_params)
            arima_model = self._model_selection_function(x_train, y_train, func_to_get_model)
            self._model = arima_model._model

    @classmethod
    def _func_to_generate_model(cls, order_params):
        def func_to_generate():
            return class_(order_params)
        class_ = cls
        return func_to_generate

    def predict(self, x, n_samples):
        if not self._is_fit():
            raise ValueError("Fit must be executed before predict.")
        if x.ndim > 2 and x.shape[2] == 1:
            x = x[:, :, 0]
        if x.ndim == 1:
            return self._predict_n_samples(x, n_samples)
        elif x.ndim == 2:
            return numpy.apply_along_axis(lambda m: self._predict_n_samples(m, n_samples), 1, x)
        else:
            raise ValueError("x values has wrong dimensions")

    def _predict_n_samples(self, instance, n_samples):
        exp_model = self._model.apply(instance)  # Use fit model with new instance
        start_index = instance.shape[0]
        end_index = start_index + n_samples - 1
        return exp_model.predict(start=start_index, end=end_index)
