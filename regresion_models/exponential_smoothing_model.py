import numpy
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from .exponential_smoothing_function import ExponentialSmoothingFunction


class ExponentialSmoothingModel(object):
    def __init__(self, exponential_params, model_selection_function=None, fit_params=None):
        self._exponential_params = exponential_params
        self._fit_params = fit_params  # Si no es None no se optimiza
        self._func_to_get_model = None
        # A esta funcion le pasas (X_train, y_train, exponential_params)
        self._model_selection_function = model_selection_function  # Puede ser None?
        self._instance_window_size = None
        self._number = numpy.random.randint(500)

    @property
    def instance_window_size(self):
        return self._instance_window_size

    def __str__(self):
        return "ExpSmoothingModel(%s)" % self.params.__str__()

    @property
    def params(self):
        return self._exponential_params

    @property
    def optimized_params(self):
        return self._fit_params  # Diccionario con la info del modelo seleccionado?

    @classmethod
    def _return_model_function(cls, exponential_params, fit_params=None):
        func_to_get_model = ExponentialSmoothingFunction.internal_fit_exponential_func(exponential_params, fit_params)
        return func_to_get_model

    def _is_fit(self):
        return self._func_to_get_model is not None

    def fit(self, x_train, y_train):
        # print("fit", self._model_selection_function, self._number)
        if self._is_fit():
            raise ValueError("Fit was already performed in this model.")
        new_model = self.__class__(self._exponential_params, model_selection_function=self._model_selection_function,
                                   fit_params=self._fit_params)
        new_model._fit(x_train, y_train)
        return new_model

    def _fit(self, x_train, y_train):
        fit_params = self._fit_params
        if fit_params is None:
            fit_params = self._get_fit_params(x_train, y_train)
        self._save_fit_elements(fit_params)

    def _get_fit_params(self, x_train, y_train):
        # print("get_fit", self._model_selection_function, self._number)
        # print(x_train.shape)
        if x_train.shape[0] < 1:
            raise ValueError("x_train must be a matrix with some instances.")
        if x_train.ndim == 1:
            get_model_func = self._return_model_function(self._exponential_params, self._fit_params)
            model = get_model_func(x_train)
            return ExponentialSmoothingFunction.return_exponential_model_params(model)
        self._instance_window_size = x_train.shape[1]
        fit_params = self._fit_params_model_selection(x_train, y_train)
        return fit_params

    def _fit_params_model_selection(self, x_train, y_train):
        if self._model_selection_function is None:
            raise ValueError("model_selection_function is None, so fit model can not be trained with a matrix.")
        func_to_get_model = self._func_to_generate_model(self._exponential_params, self._fit_params)
        class_model = self._model_selection_function(x_train, y_train, func_to_get_model)
        return class_model.optimized_params

    @classmethod
    def _func_to_generate_model(cls, exponential_params, fit_params=None):
        def func_to_generate():
            return class_(exponential_params, fit_params=fit_params)
        class_ = cls
        return func_to_generate

    def _save_fit_elements(self, fit_params):
        self._fit_params = fit_params
        self._func_to_get_model = self._return_model_function(self._exponential_params, fit_params)

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
        exp_model = self.return_instance_exp_model(instance)  # Returns ExponentialSmoothing model
        return exp_model.forecast(n_samples)

    def return_instance_exp_model(self, instance):
        # Returns ExponentialSmoothing model
        exp_model = self._func_to_get_model(instance)
        return exp_model

    @classmethod
    def _read_fit_params(cls, exp_model):
        fit_params = ExponentialSmoothingFunction.return_exponential_model_params(exp_model)
        return fit_params

    @classmethod
    def _copy_model(cls, instance, exp_model, exponential_params):
        fit_params = ExponentialSmoothingFunction.return_exponential_model_params(exp_model)
        func_to_get_model = ExponentialSmoothingFunction.internal_fit_exponential_func(exponential_params, fit_params)
        return func_to_get_model(instance)
