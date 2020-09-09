import pandas
import numpy
from window_functions.windowing import Windowing


class SeriesForecasting(object):

    @classmethod
    def forecast_time_series(cls, time_series, model, start_date=None, end_date=None, step=None):
        # time_series must be a pandas DataFrame or Series with dates in the index
        # start_date and end_date must be inside the dates period
        index_dates = time_series.index
        start_date, end_date, step = cls._adapt_forecasting_inputs(index_dates, start_date, end_date, step, model.instance_window_size)
        data_dates = cls._return_data_dates(index_dates, start_date, end_date, step, model.instance_window_size)
        forecast_series = cls._forecast_selected_time_series(time_series.loc[data_dates], model, step)
        final_dates = cls._return_output_dates(index_dates, start_date, end_date)
        return cls._adapt_output_time_series(forecast_series, final_dates, time_series)

    @classmethod
    def selected_time_period(cls, time_series, start_date=None, end_date=None, step=None, instance_window_size=0):
        index_dates = time_series.index
        start_date, end_date, step = cls._adapt_forecasting_inputs(index_dates, start_date, end_date, step, instance_window_size)
        data_dates = cls._return_data_dates(index_dates, start_date, end_date, step, instance_window_size)
        return time_series.loc[data_dates]

    @classmethod
    def forecast_time_period(cls, time_series, start_date=None, end_date=None, step=None, instance_window_size=0):
        index_dates = time_series.index
        start_date, end_date, step = cls._adapt_forecasting_inputs(index_dates, start_date, end_date, step, instance_window_size)
        final_dates = cls._return_output_dates(index_dates, start_date, end_date)
        return time_series.loc[final_dates]

    @classmethod
    def _forecast_selected_time_series(cls, time_series, model, step):
        data = time_series.to_numpy()
        windowing_function = Windowing.function_windowing(model.instance_window_size, step)
        instances = windowing_function(data)
        forecast_instances = model.predict(instances[:-1, :, :], step)  # The last one is not necessary
        if forecast_instances.ndim > 2:
            return forecast_instances.reshape(-1, forecast_instances.shape[2])
        else:
            return forecast_instances.reshape(-1, 1)

    @classmethod
    def _adapt_forecasting_inputs(cls, index_dates, start_date, end_date, step, window_size=0):
        start_date = start_date if start_date is not None else index_dates[window_size]
        end_date = end_date if end_date is not None else index_dates[-1]
        step = step if step is not None else window_size
        step = step if step != 0 else 1
        return start_date, end_date, step

    @classmethod
    def _adapt_output_time_series(cls, forecast_series, final_dates, time_series=None):
        if isinstance(time_series, pandas.Series):
            return pandas.Series(forecast_series[:len(final_dates), 0], index=final_dates, name=time_series.name)
        if time_series is not None:
            return pandas.DataFrame(forecast_series[:len(final_dates), :], index=final_dates, columns=time_series.columns)
        return pandas.DataFrame(forecast_series[:len(final_dates), :], index=final_dates)

    @classmethod
    def _adapt_time_series(cls, time_series):
        if isinstance(time_series, pandas.Series):
            time_series = pandas.DataFrame(time_series)
        return time_series

    @classmethod
    def _return_output_dates(cls, data_dates, start_date, end_date):
        return data_dates[data_dates.get_loc(start_date):data_dates.get_loc(end_date)]

    @classmethod
    def _check_time_series(cls, time_series, model):
        if not isinstance(time_series, pandas.DataFrame):
            raise ValueError("time_series must be a pandas DataFrame.")
        if len(time_series.columns) != model.num_vars:
            raise ValueError("Var number does not match the trained model.")

    @staticmethod
    def return_relative_date(index_dates, start_date, relative):
        return index_dates[index_dates.get_loc(start_date) + relative]

    @classmethod
    def _return_data_dates(cls, index_dates, start_date, end_date, step, model_window=0):
        real_end_date = cls.return_relative_date(index_dates, end_date, 0)
        n_dates = int(numpy.ceil((index_dates.get_loc(real_end_date) - index_dates.get_loc(start_date)) / step) * step)
        final_date = cls.return_relative_date(index_dates, start_date, n_dates)
        real_start_date = cls.return_relative_date(index_dates, start_date, -model_window)
        return index_dates[index_dates.get_loc(real_start_date):index_dates.get_loc(final_date)]


