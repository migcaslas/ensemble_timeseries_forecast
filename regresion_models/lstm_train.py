from sklearn.model_selection import train_test_split
from .select_best_models import SelectBestModel
import numpy
import keras


class LSTM_train(object):

    @classmethod
    def function_cumulative_train(cls, model_size=1, random_state=None):
        def func(x_matrix, y_matrix, generate_model_func):
            x2train, y2train = x_matrix, y_matrix
            if model_size < 1:
                _, x2train, _, y2train = train_test_split(x_matrix, y_matrix, test_size=model_size, random_state=random_state)
            selected_model = generate_model_func()
            for N in range(x2train.shape[0]):
                selected_model.fit(x2train[N, :], y2train[N, :])
                if any(numpy.isnan(selected_model.predict(x2train[N, :], y_matrix.shape[1]))):
                    print("NaN in model!")
                    selected_model._LSTM_model = keras.models.load_model('temp_model.h5')
                else:
                    selected_model._LSTM_model.save('temp_model.h5')
            print("Selected model", selected_model)
            print("Selected model", selected_model._LSTM_model)
            return selected_model
        return func