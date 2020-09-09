import numpy
from statsmodels.tsa.arima.model import ARIMA


class ArimaDinModel(object):
    def __init__(self, order_params, model_selection_function=None):
        self._order_params = order_params
        # A esta funcion le pasas (X_train, y_train, exponential_params)
        self._model_selection_function = model_selection_function  # Puede ser None?
        self._instance_window_size = None

    @property
    def instance_window_size(self):
        return self._instance_window_size

    def __str__(self):
        return "ArimaDinModel(%s)" % self.params.__str__()

    @property
    def params(self):
        return self._order_params

    @classmethod
    def _return_model_function(cls, order_params):
        def fun(instance, y_values=None):
            return ARIMA(instance, order=order_params)
        return fun

    def _is_fit(self):
        return True

    def fit(self, x_train, y_train):
        return self.__class__(self._order_params, model_selection_function=self._model_selection_function)

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
        exp_model = ARIMA(instance, order=self._order_params).fit()  # Use fit model with new instance
        start_index = instance.shape[0]
        end_index = start_index + n_samples - 1
        return exp_model.predict(start=start_index, end=end_index)
