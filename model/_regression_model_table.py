import pandas


class _RegressionModelTable(object):
    def __init__(self, regression_models, function_to_evaluate_model=None, function_to_select_model=None):

        if not isinstance(regression_models, list):
            regression_models = [regression_models]

        self._check_model_inputs(regression_models, function_to_evaluate_model, function_to_select_model)

        self._function_to_evaluate_model = function_to_evaluate_model
        self._function_to_select_model = function_to_select_model

        self._regression_model_list = regression_models
        self._table_evaluation_dict = {}
        self._fit_model_table_dict = {}

    @property
    def pandas_table(self):
        model_names = [model.__str__() for model in self._regression_model_list]
        df = pandas.DataFrame(self._table_evaluation_dict, index=model_names)
        df = df.transpose()
        return df

    @classmethod
    def _check_model_inputs(cls, regression_models, function_to_evaluate_model, function_to_select_model):
        if len(regression_models) > 1:
            if function_to_select_model is None or function_to_evaluate_model is None:
                raise ValueError("Functions to evaluate and select regression models must be specified "
                                 "in case of regression model list.")

    def initialize_tables(self, label_names):
        n_models = len(self._regression_model_list)
        self._table_evaluation_dict = {LABEL_NAME: [None]*n_models for LABEL_NAME in label_names}
        self._fit_model_table_dict = {LABEL_NAME: None for LABEL_NAME in label_names}

    def evaluate_label_models(self, x, y, label_name):
        label_evaluation_list = list(map(lambda model: self.evaluate_model(model, x, y), self._regression_model_list))
        self._table_evaluation_dict[label_name] = label_evaluation_list

    def evaluate_model(self, model, x, y):
        return self._function_to_evaluate_model(model, x, y)

    def return_selected_label_model(self, label_name):
        if len(self._regression_model_list) == 1:
            print("unique model")
            return self._regression_model_list[0]
        if self._is_any_none_in_list(self._table_evaluation_dict[label_name]):
            raise ValueError("Some models were not evaluated")
        return self._function_to_select_model(self._regression_model_list, self._table_evaluation_dict[label_name])

    @staticmethod
    def _is_any_none_in_list(list_):
        return any(list(map(lambda x: x is None, list_)))

    def set_label_regression_model(self, model, label_name):
        self._fit_model_table_dict[label_name] = model

    def return_label_regression_model(self, label_name):
        return self._fit_model_table_dict[label_name]

    @classmethod
    def _predict_func(cls, model, x_instance, n_samples):
        return model.predict(x_instance, n_samples)
