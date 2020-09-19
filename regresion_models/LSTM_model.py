import numpy
import keras


class LSTMModel(object):
    def __init__(self, LSTM_model, win_param, fit_params, opt_params, model_selection_function=None):
        self._LSTM_model = LSTM_model
        self._win_param = win_param
        self._fit_params = fit_params
        self._opt_params = opt_params
        self._model = None
        # A esta funcion le pasas (X_train, y_train, exponential_params)
        self._model_selection_function = model_selection_function  # Puede ser None?
        self._batch_size = None

    @property
    def instance_window_size(self):
        return self._instance_window_size

    # def __str__(self):
    #     return "LSTMModel(num_layers=%s)" % str(len(self._LSTM_model.layers))

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
        fit_model = self._fit(x_train, y_train)
        return fit_model

    def _fit(self, x_train, y_train):
        print(self._LSTM_model)
        if x_train.shape[0] < 1:
            raise ValueError("x_train must be a matrix with some instances.")
        if x_train.ndim == 1:
            adp_x, adp_y = self._split_sequence(numpy.concatenate((x_train, y_train)), self._win_param, len(y_train))
            adp_x = adp_x.reshape((adp_x.shape[0], adp_x.shape[1], 1))
            self._LSTM_model.fit(adp_x, adp_y, **self._fit_params)
            print("ndim=1", self._LSTM_model)
            return self
        else:
            self._instance_window_size = x_train.shape[1]
            func_to_get_model = self._func_to_generate_model(self._LSTM_model, self._win_param,
                                                             self._fit_params, self._opt_params)
            LSTM_model = self._model_selection_function(x_train, y_train, func_to_get_model)
            # copy_model = keras.models.clone_model(LSTM_model._LSTM_model)
            # copy_model.set_weights(LSTM_model._LSTM_model.get_weights())
            # copy_model.compile(**self._opt_params)
            fit_model = func_to_get_model()
            fit_model._model = LSTM_model._LSTM_model
            print("ndim=2", fit_model, fit_model._model)
            return fit_model

    @classmethod
    def _func_to_generate_model(cls, LSTM_model, win_parm, fit_params, opt_params):
        def func_to_generate():
            copy_model = keras.models.clone_model(LSTM_model)
            copy_model.set_weights(LSTM_model.get_weights())
            copy_model.compile(**opt_params)
            return class_(copy_model, win_parm, fit_params, opt_params)
        class_ = cls
        return func_to_generate

    def predict(self, x, n_samples):
        if x.ndim > 2 and x.shape[2] == 1:
            x = x[:, :, 0]
        if x.ndim == 1:
            adp_instance = x[-self._win_param:].reshape(1, -1, 1)
            forecast = self._LSTM_model.predict(adp_instance)[0, :]
            if len(forecast) > n_samples:
                raise ValueError("Num of samples is greater than the output of the neural network")
            return forecast[:n_samples]
        elif x.ndim == 2:
            if not self._is_fit():
                raise ValueError("Fit must be executed before predict.")
            adp_instances = x[:, -self._win_param:]
            forecast_matrix = numpy.apply_along_axis(lambda m: self._predict_n_samples(m), 1, adp_instances)
            if forecast_matrix.shape[1] > n_samples:
                raise ValueError("Num of samples is greater than the output of the neural network")
            return forecast_matrix[:, :n_samples]
        else:
            raise ValueError("x values has wrong dimensions")

    def _predict_n_samples(self, instance):
        adp_instance = instance.reshape(1, -1, 1)
        forecast = self._model.predict(adp_instance)[0, :]
        return forecast

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
