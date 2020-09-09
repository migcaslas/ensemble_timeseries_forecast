import numpy


class LSTMModel(object):
    def __init__(self, LSTM_model, win_param, fit_params, model_selection_function=None):
        self._LSTM_model = LSTM_model
        self._win_param = win_param
        self._fit_params = fit_params
        self._model = None
        # A esta funcion le pasas (X_train, y_train, exponential_params)
        self._model_selection_function = model_selection_function  # Puede ser None?
        self._batch_size = None

    @property
    def instance_window_size(self):
        return self._instance_window_size

    def __str__(self):
        return "LSTMModel(num_layers=%s)" % str(len(self._LSTM_model.layers))

    @property
    def params(self):
        return self._fit_params

    @classmethod
    def _return_model_function(cls, model, fit_params):
        def fun(instance, y_values=None):
            return model(instance, y_values, **fit_params)
        return fun

    def _is_fit(self):
        return self._model is not None

    def fit(self, x_train, y_train):
        if self._is_fit():
            raise ValueError("Fit was already performed in this model.")
        self._fit(x_train, y_train)
        return self

    def _fit(self, x_train, y_train):
        if x_train.shape[0] < 1:
            raise ValueError("x_train must be a matrix with some instances.")
        if x_train.ndim == 1:
            adp_x, adp_y = self._split_sequence(numpy.concatenate((x_train, y_train)), self._win_param, len(y_train))
            adp_x = adp_x.reshape((adp_x.shape[0], adp_x.shape[1], 1))
            self._LSTM_model.fit(adp_x, adp_y, **self._fit_params)
        else:
            self._instance_window_size = x_train.shape[1]
            func_to_get_model = self._func_to_generate_model(self._LSTM_model, self._win_param, self._fit_params)
            LSTM_model = self._model_selection_function(x_train, y_train, func_to_get_model)
            self._model = LSTM_model._LSTM_model

    @classmethod
    def _func_to_generate_model(cls, LSTM_model, win_parm, fit_params):
        def func_to_generate():
            return class_(LSTM_model, win_parm, fit_params)
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
        adp_instance = instance[-self._win_param:].reshape(1, -1, 1)
        forecast = self._model.predict(adp_instance)[0, :]
        if len(forecast) > n_samples:
            raise ValueError("Num of samples is greater than the output of the neural network")
        return forecast[0:n_samples]

    @staticmethod
    def _split_sequence(sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return numpy.array(X), numpy.array(y)
