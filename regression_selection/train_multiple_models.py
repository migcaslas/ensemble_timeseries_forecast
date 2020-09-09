from sklearn.model_selection import train_test_split
from .select_best_models import SelectBestModel


class TrainMultipleModels(object):

    @classmethod
    def function_cumulative_train(cls, model_size=1, random_state=None):
        def func(x_matrix, y_matrix, generate_model_func):
            x2train, y2train = x_matrix, y_matrix
            if model_size < 1:
                _, x2train, _, y2train = train_test_split(x_matrix, y_matrix, test_size=model_size, random_state=random_state)
            selected_model = generate_model_func()
            for N in range(x2train.shape[0]):
                selected_model.fit(x2train[N, :], y2train[N, :])
            print("Selected model", selected_model)
            return selected_model
        return func

    @classmethod
    def function_best_model_on_segment(cls, evaluate_func, selection_func, model_size=1, test_size=1, random_state=None):
        def func(x_matrix, y_matrix, generate_model_func):
            x2train, y2train = x_matrix, y_matrix
            if model_size < 1:
                _, x2train, _, y2train = train_test_split(x_matrix, y_matrix, test_size=model_size, random_state=random_state)
            x2test, y2test = x2train, y2train
            if test_size < 1:
                _, x2test, _, y2test = train_test_split(x_matrix, y_matrix, test_size=test_size, random_state=random_state)
            models = [generate_model_func().fit(x2train[N, :], y2train[N, :]) for N in range(x2train.shape[0])]
            selected_model = select_best_model_func(models, x2test, y2test, cls._predict_func)
            print("Best model", selected_model)
            return selected_model
        select_best_model_func = SelectBestModel.function_to_select_best_model(evaluate_func, selection_func)
        return func

    @classmethod
    def _predict_func(cls, model, x_instance, n_samples):
        return model.predict(x_instance, n_samples)
