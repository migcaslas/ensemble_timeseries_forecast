from .windowing import Windowing


class WindowingData(object):

    @classmethod
    def window_data_predictions(cls, data, window_samples, forecasting_horizon_samples, step_samples=1):
        if data.ndim > 2:
            raise ValueError("Data array can not have more than two dimensions (times, variables)")
        total_window_size = window_samples + forecasting_horizon_samples
        windowing_function = Windowing.function_windowing(total_window_size, step_samples)
        windowed_data = windowing_function(data)
        window_data = windowed_data[:, 0:window_samples, :]
        prediction_data = windowed_data[:, window_samples:, :]
        return window_data, prediction_data
