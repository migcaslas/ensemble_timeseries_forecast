import pandas
import numpy
import warnings

from window_functions.windowing_data import WindowingData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model.ensemble_model import EnsembleModel
from sklearn.cluster import KMeans

from regression_selection.train_multiple_models import TrainMultipleModels
from regression_selection.select_best_models import SelectBestModel
from regression_selection.evaluate_model import EvaluateModel
from regression_funcs.error_computation import ErrorComputation

from regresion_models.arima_model import ArimaModel
from regresion_models.exponential_smoothing_model import ExponentialSmoothingModel
from regresion_models.arima_din_model import ArimaDinModel

from sklearn import preprocessing
from model_evaluation.model_graphs import ModelGraphs


# Adapt function cluster
def adapt_data_for_clustering(x_data):
    return numpy.stack((numpy.mean(x_data, axis=1), numpy.max(x_data, axis=1) - numpy.min(x_data, axis=1)), axis=1)


def split_data(pandas_data_, end):
    dates_ = pandas_data_.index
    train_dates_ = dates_[:dates_.get_loc(end)]
    test_dates_ = dates_[dates_.get_loc(end):]
    return pandas_data_.loc[train_dates_], pandas_data_.loc[test_dates_]


# Read data
start_date = '2013-01-01 00:00:00'
data_ds = pandas.read_pickle("univariante.pkl")
data_ds.name = "consumption"
# pandas_train, pandas_test = split_data(data_ds, start_date)

min_max_scaler = preprocessing.MinMaxScaler()
scaled_data = min_max_scaler.fit_transform(data_ds.values.reshape(-1, 1))
scaled_data_ds = pandas.Series(scaled_data.reshape(-1), index=data_ds.index)
pandas_train, pandas_test = split_data(scaled_data_ds, start_date)

W = 24 * 8
H = 24
S = 24

# Data to matrix
X_train, y_train = WindowingData.window_data_predictions(pandas_train.to_numpy(), W, H, S)  # X windowed data, y prediction
X_test, y_test = WindowingData.window_data_predictions(pandas_test.to_numpy(), W, H, S)  # X windowed data, y prediction

# Cluster class
kmeans = KMeans(n_clusters=7, random_state=42)

# Classifier
# clf = SVC(kernel='rbf', C=1, gamma=0.01)
# clf.fit(Xt, yt)
# y_pred = clf.predict(Xtest)

# Return Regression model
mape_func = ErrorComputation.MAPE
select_min_func = SelectBestModel.function_to_select_minimum_mean_error
model_selection_func = TrainMultipleModels.function_best_model_on_segment(mape_func, select_min_func, model_size=0.01, test_size=0.01)

exp_model_params = {"seasonal": "multiplicative", "seasonal_periods": 24*6}
exp_day_mult_reg_model = ExponentialSmoothingModel(exponential_params=exp_model_params, model_selection_function=model_selection_func)

reg_models = [exp_day_mult_reg_model, ArimaModel((3, 0, 1), model_selection_func),
              ArimaModel((5, 0, 1), model_selection_func), ArimaModel((7, 0, 1), model_selection_func)]

reg_model_evaluate_func = EvaluateModel.split_evaluate_function_model(test_size=0.33, evaluate_func=ErrorComputation.MAPE)
# reg_model_evaluate_func = EvaluateModel.split_evaluate_function(test_size=0.0033, evaluate_func=ErrorComputation.MAPE)
reg_model_selection_func = EvaluateModel.select_min_error_model

ensemble_model = EnsembleModel(kmeans, reg_models,
                               function_to_evaluate_model=reg_model_evaluate_func,
                               function_to_select_model=reg_model_selection_func)

ensemble_model = ensemble_model.fit(X_train, y_train)
forecast_scaled = ensemble_model.predict(X_test, H)

forecast_y = numpy.apply_along_axis(lambda x: min_max_scaler.inverse_transform(x.reshape(1, -1)).reshape(-1), arr=forecast_scaled, axis=1)
y_test_rescaled = numpy.apply_along_axis(lambda x: min_max_scaler.inverse_transform(x.reshape(1, -1)).reshape(-1), arr=y_test, axis=1)

mape = ErrorComputation.MAPE_instances(forecast_y, y_test_rescaled)
rmse = ErrorComputation.RMSE_instances(forecast_y, y_test_rescaled)
print("RMSE: (media: %s; std:%s)" % (numpy.mean(rmse), numpy.std(rmse)))
print("MAPE: (media: %s; std:%s)" % (numpy.mean(mape), numpy.std(mape)))

########################################################################################################################

###

# start_date = '2011-07-01 00:00:00'
# end_date = '2011-01-01 00:00:00'
end_date = '2013-07-01 00:00:00'
# end_date = data_ds.index[-1]

forecast_output = ModelGraphs.predicted_data_graph(scaled_data_ds, ensemble_model, start_date, end_date=end_date, step=S)
# ModelGraphs.plot_time_series_labelled(scaled_data_ds, ensemble_model, S)
ensemble_model.model_regression_table.plot.bar(rot=0)

# self = ensemble_model._regression_model_table
