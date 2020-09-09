import numpy


class DisplayError(object):

    @classmethod
    def display_multivariate_error(cls, y_test, forecast_y, function_to_evaluate, error_name=None):
        output = []
        for VAR_IDX in range(y_test.shape[2]):
            error_values = function_to_evaluate(forecast_y[:, :, VAR_IDX], y_test[:, :, VAR_IDX])
            output.append(cls.display_error_msg(error_values, name="VAR_"+str(VAR_IDX), error_name=error_name))
        return output

    @staticmethod
    def display_error_msg(error_values, name="", error_name=None):
        error_name = error_name if error_name is not None else "Error"
        error_values[numpy.isinf(error_values)] = numpy.nan
        mean_value = 100 * numpy.nanmean(error_values)
        std_value = 100 * numpy.nanstd(error_values)
        str_ = "%s %s: (media: %.5f%%; std:%.5f%%)" % (error_name, name, mean_value, std_value)
        print(str_)
        return str_

    @classmethod
    def display_univariate_error(cls, y_test, forecast_y, function_to_evaluate):
        y_test = cls._adapt_univariate_var(y_test)
        forecast_y = cls._adapt_univariate_var(forecast_y)
        error_values = function_to_evaluate(forecast_y, y_test)
        cls.display_error_msg(error_values, "VAR")

    @classmethod
    def compute_univariate_model(cls, model, x_test, y_test):
        cls._check_x_y_dim(x_test, y_test)
        x_test, y_test = cls._adapt_univariate_inputs(x_test, y_test)
        return model.predict(x_test, y_test.shape[1])

    @classmethod
    def _adapt_univariate_inputs(cls, x_test, y_test):
        x_test = cls._adapt_univariate_var(x_test)
        y_test = cls._adapt_univariate_var(y_test)
        return x_test, y_test

    @classmethod
    def _adapt_univariate_var(cls, matrix_test):
        if matrix_test.ndim > 2:
            if matrix_test.shape[2] > 1:
                raise ValueError("Univarate analysis can not be performed on multivariate matrix")
            return matrix_test[:, :, 0]
        return matrix_test

    @classmethod
    def _check_x_y_dim(cls, x_test, y_test):
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError("Both matrix must have the same number of instances")
        if x_test.ndim != y_test.ndim:
            raise ValueError("Both matrix must have the same number of dimensions")
        if x_test.ndim > 2:
            if x_test.shape[2] != y_test.shape[2]:
                raise ValueError("Both matrix must have the same number of variables")
