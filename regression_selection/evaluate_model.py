import numpy
from sklearn.model_selection import train_test_split


class EvaluateModel(object):

    @classmethod
    def split_evaluate_function(cls, test_size,  evaluate_func, func_to_fit=None, func_to_predict=None, **kwargs):
        def split_func(x_matrix, y_matrix):
            return train_test_split(x_matrix, y_matrix, test_size=test_size, **kwargs)
        return cls.return_default_evaluate_function(split_func, evaluate_func, func_to_fit=func_to_fit, func_to_predict=func_to_predict)

    @classmethod
    def split_evaluate_function_model(cls, test_size,  evaluate_func, func_to_fit=None, func_to_predict=None, **kwargs):
        def split_func(x_matrix, y_matrix):
            return train_test_split(x_matrix, y_matrix, test_size=test_size, **kwargs)
        return cls.return_default_evaluate_function_model(split_func, evaluate_func, func_to_fit=func_to_fit, func_to_predict=func_to_predict)

    @classmethod
    def select_min_error_model(cls, models, errors):
        idx = numpy.argmin(errors)
        return models[idx]

    @classmethod
    def return_default_evaluate_function(cls, func_to_split, func_to_evaluate, func_to_fit=None, func_to_predict=None):
        def func(model, x_matrix, y_matrix):
            n_instances, n_samples = y_matrix.shape
            x_train, x_test, y_train, y_test = func_to_split(x_matrix, y_matrix)
            fit_model = func_to_fit(model, x_train, y_train)
            predicted_y = func_to_predict(fit_model, x_test, n_samples)
            return func_to_evaluate(predicted_y, y_test)
        func_to_fit = func_to_fit if func_to_fit is not None else cls.default_model_fit_function
        func_to_predict = func_to_predict if func_to_predict is not None else cls.default_model_predict_function
        return func

    @classmethod
    def return_default_evaluate_function_model(cls, func_to_split, func_to_evaluate, func_to_fit=None, func_to_predict=None):
        def func(model, x_matrix, y_matrix):
            n_instances, n_samples = y_matrix.shape
            x_train, x_test, y_train, y_test = func_to_split(x_matrix, y_matrix)
            fit_model = func_to_fit(model, x_train, y_train)
            # print("EVALUATE", model)
            # print("EVALUATE", fit_model)
            # print("EVALUATE", fit_model._model)
            predicted_y = func_to_predict(fit_model, x_test, n_samples)
            return fit_model, func_to_evaluate(predicted_y, y_test)
        func_to_fit = func_to_fit if func_to_fit is not None else cls.default_model_fit_function
        func_to_predict = func_to_predict if func_to_predict is not None else cls.default_model_predict_function
        return func

    @classmethod
    def default_model_fit_function(cls, model, x, y):
        return model.fit(x, y)

    @classmethod
    def default_model_predict_function(cls, model, x, n_samples):
        return model.predict(x, n_samples)
