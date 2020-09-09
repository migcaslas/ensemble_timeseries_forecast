import numpy
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split


class ExponentialSmoothingFunction(object):

    @classmethod
    def internal_fit_exponential_func(cls, exponential_params, fit_params=None):
        def int_fit_function(instance, y_values=None):
            # instance: array len=W
            return ExponentialSmoothing(instance, **exponential_params).fit(**fit_params)
        fit_params = fit_params if fit_params is not None else {}
        return int_fit_function

    @classmethod
    def return_exponential_model_params(cls, model):
        param_names = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'damping_slope']
        return {PARAM_NAME: model.params[PARAM_NAME] for PARAM_NAME in param_names}

    @classmethod
    def return_exponential_forecasting_function(cls, exponential_params, model_params=None):
        def compute_forecast(instance, n_samples):
            fit_func = cls.internal_fit_exponential_func(exponential_params, model_params)
            model = fit_func(instance)
            return model.forecast(n_samples)
        return compute_forecast

    @classmethod
    def compute_model_rse_array(cls, train_instance, X, y, exponential_params):
        model_function = cls.internal_fit_exponential_func(exponential_params)
        model = model_function(train_instance)
        params = cls.return_exponential_model_params(model)
        forecasting_function = cls.return_exponential_forecasting_function(exponential_params, params)
        n_samples = y.shape[1]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        forecasted_y = numpy.apply_along_axis(lambda x: forecasting_function(x, n_samples), 1, X_test)
        return numpy.nanmean(numpy.abs(y_test - forecasted_y) / y_test, axis=1)

    @classmethod
    def compute_best_model(cls, X, y, exponential_params):
        _, X_test, _, _ = train_test_split(X, y, test_size=0.1, random_state=42)
        rse = numpy.apply_along_axis(lambda x: cls.compute_model_rse_array(x, X, y, exponential_params), 1, X_test)
        rse_mean = numpy.mean(rse, axis=1)
        idx = numpy.argmin(rse_mean)
        model_func = cls.internal_fit_exponential_func(exponential_params)
        return model_func(X[idx, :])

    @classmethod
    def return_exponential_model_selection_function(cls, exponential_params):
        def return_modeling_function(X, y):
            model = cls.compute_best_model(X, y, exponential_params)
            best_params = cls.return_exponential_model_params(model)
            return cls.return_exponential_forecasting_function(exponential_params, best_params)
        return return_modeling_function
