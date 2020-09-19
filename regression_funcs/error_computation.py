import numpy


class ErrorComputation(object):

    @classmethod
    def MAE_instances(cls, forecast_y, y_test):
        return numpy.nanmean(cls._absolute_error(forecast_y, y_test), axis=1)

    @classmethod
    def MAE(cls, forecast_y, y_test):
        return numpy.nanmean(cls._absolute_error(forecast_y, y_test))

    @classmethod
    def _absolute_error(cls, forecast_y, y_test):
        return numpy.abs(y_test - forecast_y)

    @classmethod
    def RMSE_instances(cls, forecast_y_matrix, y_test_matrix):
        return numpy.sqrt(numpy.nanmean(cls._absolute_error(forecast_y_matrix, y_test_matrix)**2, axis=1))

    @classmethod
    def RMSE(cls, forecast_y, y_test):
        return numpy.sqrt(numpy.nanmean(cls._absolute_error(forecast_y, y_test)**2))

    @classmethod
    def MAPE_instances(cls, forecast_y_matrix, y_test_matrix):
        return numpy.nanmean(cls._percentage_error(forecast_y_matrix, y_test_matrix), axis=1)

    @classmethod
    def MAPE(cls, forecast_y, y_test):
        return numpy.nanmean(cls._percentage_error(forecast_y, y_test))

    @classmethod
    def _percentage_error(cls, forecast_y, y_test):
        return numpy.abs(y_test - forecast_y) / y_test

