import numpy
from sklearn.model_selection import train_test_split


class SelectBestModel(object):

    @classmethod
    def function_to_select_best_model(cls, evaluate_func, selection_func):
        def func(models, x_matrix, y_matrix, func_to_predict):
            n_instances, n_samples = y_matrix.shape
            models_matrix = numpy.zeros((len(models), n_instances))
            for IDX, M in enumerate(models):
                models_matrix[IDX, :] = cls.evaluate_model(M, x_matrix, y_matrix, evaluate_func, func_to_predict)
            selected_model = selection_func(models, models_matrix)
            print("\tComputed best params for: %s" % selected_model)
            return selected_model
        return func

    @classmethod
    def evaluate_model(cls, model, x_matrix, y_matrix, evaluate_func, func_to_predict):
        n_instances, n_samples = y_matrix.shape
        model_array = numpy.zeros(n_instances)
        for R in range(n_instances):
            predict_y = func_to_predict(model, x_matrix[R, :], n_samples)
            model_array[R] = evaluate_func(predict_y, y_matrix[R, :])
        return model_array

    @classmethod
    def function_to_select_minimum_mean_error(cls, models, models_errors):
        model_means = numpy.mean(models_errors, axis=1)
        min_idx = numpy.argmin(model_means)
        return models[min_idx]
