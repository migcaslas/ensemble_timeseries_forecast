import numpy


class Windowing(object):

    @classmethod
    def function_windowing(cls, WINDOW, STEP=1):
        # data is a numpy matrix: (times, variables)
        # WINDOW must be multiple of STEP
        # Output matrix: (L-W)/S, W, V)
        def return_function(data):
            if data.ndim > 2:
                raise ValueError("Data array can not have more than two dimensions (times, variables)")
            if data.ndim == 1:
                data = data.reshape((-1, 1))
            return cls._window_series(data, WINDOW, STEP=STEP)
        cls._check_window_size_multiple(WINDOW, STEP)
        return return_function

    @classmethod
    def _window_series(cls, data, WINDOW, STEP=1):
        # data must be a numpy matrix: (times, variables)
        # WINDOW must be multiple of STEP
        cls._check_window_size_multiple(WINDOW, STEP)
        LONG, VARS = data.shape
        num_rows = int(numpy.floor((LONG - WINDOW) / STEP)) + 1
        window_data = numpy.zeros((num_rows, WINDOW, VARS))
        for START in range(int(numpy.floor(WINDOW / STEP))):
            start_idx = START * STEP
            num_mat = int(numpy.floor((LONG - start_idx) / WINDOW))
            last_idx = WINDOW * num_mat + start_idx
            adp_matrix = data[start_idx:last_idx, :].reshape((-1, WINDOW, VARS))
            idx_array = numpy.arange(start_idx / STEP, last_idx / STEP, WINDOW / STEP, dtype=int)
            window_data[idx_array, :, :] = adp_matrix
        return window_data

    @classmethod
    def _check_window_size_multiple(cls, window_size, step_size):
        if window_size / step_size != numpy.round(window_size / step_size):
            raise ValueError("This method only works if W is multiple of S.")
