import pandas
import numpy
import warnings
import matplotlib.pyplot as plt

from window_functions.windowing_data import WindowingData

# Load sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

# Load ensemble models
from model.ensemble_model import EnsembleModel
from model_evaluation.random_classifier import RandomClassifier

# Load evaluate and select functions
from regression_funcs.error_computation import ErrorComputation
from regression_selection.train_multiple_models import TrainMultipleModels
from regression_selection.select_best_models import SelectBestModel
from regression_selection.evaluate_model import EvaluateModel

# Load windowed models
from regresion_models.arima_model import ArimaModel
from regresion_models.exponential_smoothing_model import ExponentialSmoothingModel

# Load graph functions
from model_evaluation.model_graphs import ModelGraphs


def split_data(pandas_data_, end):
    dates_ = pandas_data_.index
    train_dates_ = dates_[:dates_.get_loc(end)]
    test_dates_ = dates_[dates_.get_loc(end):]
    return pandas_data_.loc[train_dates_], pandas_data_.loc[test_dates_]


# Filter warnings
warnings.filterwarnings("ignore")

# Load-adapt univariate data
start_date = '2013-01-01 00:00:00'
data_ds = pandas.read_csv("univariate.csv")  # 10 minutes step
data_ds = data_ds.set_index('0')['1']
data_ds.name = "consumption"
data_df = pandas.DataFrame(data_ds)
var_names = data_df.columns

# Set window params
W = 6*32  # 32 hours
H = 6*4  # 4 hours
S = 6*4  # 4 hours

# Split train-test
df_train, df_test = split_data(data_df, start_date)

# Train scaler with training_set
scalers_dict = {V: MinMaxScaler(feature_range=(0.2, 0.8)).fit(df_train[V].values.reshape(-1, 1)) for V in var_names}

# Scale
scaled_df_train = pandas.DataFrame(0, columns=var_names, index=df_train.index)
scaled_df_test = pandas.DataFrame(0, columns=var_names, index=df_test.index)
for VAR in var_names:
    scaled_df_train[VAR] = scalers_dict[VAR].transform(df_train[VAR].values.reshape(-1, 1)).reshape(-1)
    scaled_df_test[VAR] = scalers_dict[VAR].transform(df_test[VAR].values.reshape(-1, 1)) .reshape(-1)

# Data to matrix
X_sc_train, y_sc_train = WindowingData.window_data_predictions(scaled_df_train.to_numpy(), W, H, S)  # X(N, W), y(N, H)
X_sc_test, y_test_pr = WindowingData.window_data_predictions(scaled_df_test.to_numpy(), W, H, S)  # X(N_test, W), y(N_test, H)
_, y_test = WindowingData.window_data_predictions(df_test.to_numpy(), W, H, S)  # X(N_test, W), y(N_test, H)

# Get test index
_, y_idx = WindowingData.window_data_predictions(numpy.array(list(range(df_test.shape[0]))), W, H, S)
y_index = df_test.index[numpy.unique(y_idx.reshape(-1)).astype(int)]


# --- ENSEMBLE MODEL

# Cluster class
kmeans = KMeans(n_clusters=7, random_state=42)

# Classifier
clf = GaussianNB()

# Select function to select best params for windows in the group.
train_test_size_per = 0.1  # WARNING! Training and testing size is reduced for example purposes. Set to 1
MAPE_func = ErrorComputation.MAPE
select_min_func = SelectBestModel.function_to_select_minimum_mean_error
model_selection_func = TrainMultipleModels.function_best_model_on_segment(MAPE_func, select_min_func,
                                                                          model_size=train_test_size_per,
                                                                          test_size=train_test_size_per)

# Set prediction models

# ARIMA models
pred_models = [ArimaModel((3, 0, 1), model_selection_func), ArimaModel((5, 0, 1), model_selection_func),
              ArimaModel((7, 0, 1), model_selection_func)]

# Exponential SmoothingModel
# exp_model_params = {"seasonal": "multiplicative", "seasonal_periods": 24*6}
# exp_day_mult_reg_model = ExponentialSmoothingModel(exponential_params=exp_model_params,
#                                                    model_selection_function=model_selection_func)
# pred_models = [exp_day_mult_reg_model]

# Define function to select model
pred_model_evaluate_func = EvaluateModel.split_evaluate_function_model(test_size=0.33, evaluate_func=ErrorComputation.MAPE)
pred_model_selection_func = EvaluateModel.select_min_error_model

# Set ensemble model params
ensemble_model = EnsembleModel(kmeans, pred_models, classifier_model=clf,
                               function_to_evaluate_model=pred_model_evaluate_func,
                               function_to_select_model=pred_model_selection_func)

# Train model
ensemble_model = ensemble_model.fit(X_sc_train, y_sc_train)

# Forecast
forecast_scaled = ensemble_model.predict(X_sc_test, H)


# --- OUTPUTS

# Rescale
forecast = numpy.zeros(forecast_scaled.shape)
for IDX, VAR in enumerate(var_names):
    scale_func = lambda x: scalers_dict[VAR].inverse_transform(x.reshape(1, -1)).reshape(-1)
    forecast[:, :, IDX] = numpy.apply_along_axis(scale_func, arr=forecast_scaled[:, :, IDX], axis=1)

for IDX, VAR in enumerate(var_names):
    mape = ErrorComputation.MAPE_instances(forecast[:, :, IDX], y_test[:, :, IDX])
    rmse = ErrorComputation.RMSE_instances(forecast[:, :, IDX], y_test[:, :, IDX])
    mape[numpy.where(mape == numpy.Inf)] = numpy.nan
    rmse[numpy.where(rmse == numpy.Inf)] = numpy.nan
    print("RMSE %s: (media: %s; std:%s)" % (VAR, numpy.nanmean(rmse), numpy.nanstd(rmse)))
    print("MAPE %s: (media: %s; std:%s)" % (VAR, numpy.nanmean(mape), numpy.nanstd(mape)))

# Plot
forecast_df = pandas.DataFrame(0, columns=var_names, index=y_index)
actual_test_df = pandas.DataFrame(0, columns=var_names, index=y_index)
for IDX, VAR in enumerate(var_names):
    forecast_df[VAR] = forecast[:, :, IDX].reshape(-1)
    actual_test_df[VAR] = y_test[:, :, IDX].reshape(-1)

for VAR in var_names:
    plt.figure()
    actual_test_df[VAR].plot()
    ax = forecast_df[VAR].plot()
    ax.legend()

# Plot model evaluation
ensemble_model.model_regression_table.plot.bar(rot=0)

# Plot time series labelled
# ModelGraphs.plot_time_series_labelled(scaled_df_test, ensemble_model, S)
