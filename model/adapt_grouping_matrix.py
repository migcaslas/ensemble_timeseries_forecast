import numpy


class AdaptGroupingMatrix(object):

    @classmethod
    def adapt_windowed_data(cls, X, adapt_group_data_function):
        # X: ndarray, shape: (n_instances, W, V)
        # output: ndarray, shape(n_instances, K, V). Number of variables could be modified
        if adapt_group_data_function is not None:
            adp_x = adapt_group_data_function(X)
            if adp_x.shape[0] != X.shape[0]:
                raise ValueError("Error adapting data. The number of instances must be the same")
            return adp_x
        return X

    @classmethod
    def return_classification_matrix(cls, X, extra_var_dim, extra_var_matrix=None):
        # X: ndarray, shape: (n_instances, n_classification_features, n_variables)
        # extra_var_matrix: None or ndarray matrix, shape: (n_instances, n_extra_classification_features).
        # output: ndarray, shape: (n_instances, n_classification_features*n_variables + n_extra_classification_features)
        grouping_matrix = cls.adapt_to_grouping_matrix(X)
        if extra_var_dim is not None:
            grouping_matrix = cls._add_extra_var_to_grouping_matrix(grouping_matrix, extra_var_dim, extra_var_matrix)
        elif extra_var_matrix is not None:
            raise ValueError("No extra variables were added but extra_var_matrix is not None.")
        return grouping_matrix

    @staticmethod
    def adapt_to_grouping_matrix(X):
        # X: ndarray, shape: (n_instances, n_classification_features, n_variables)
        # output: ndarray, shape(n_instances, n_classification_features*n_variables)
        return numpy.moveaxis(X, 1, 2).reshape(X.shape[0], -1)

    @classmethod
    def _add_extra_var_to_grouping_matrix(cls, grouping_matrix, extra_var_dim, extra_var_matrix=None):
        # grouping_matrix: ndarray, shape: (n_instances, n_classification_features)
        # extra_var_matrix: None or ndarray matrix, shape: (n_instances, n_extra_classification_features).
        # output: ndarray, shape: (n_instances, n_classification_features + n_extra_classification_features)
        cls._check_extra_var_shape(grouping_matrix, extra_var_matrix)
        cls._check_extra_var_numbers(extra_var_matrix, extra_var_dim)
        return numpy.concatenate((grouping_matrix, extra_var_matrix), axis=1)

    @staticmethod
    def _check_extra_var_numbers(extra_var_matrix, extra_var_dim):
        if extra_var_dim is not None:
            if extra_var_matrix is None or extra_var_matrix.shape[1] != extra_var_dim:
                raise ValueError("extra_var_matrix must have same number of vars that those applied to the model.")

    @staticmethod
    def _check_extra_var_shape(X, extra_var_matrix):
        if extra_var_matrix.ndim != 2:
            raise ValueError("extra_var_matrix must have two dimensions (n_instances, n_extra_features)")
        if extra_var_matrix.shape[0] != X.shape[0]:
            raise ValueError("extra_var_matrix must have same number of instances (rows) than the X matrix.")
