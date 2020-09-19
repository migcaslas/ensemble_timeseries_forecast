----------
ENSEMBLE TIME SERIES FORECAST
----------

This algorithm combines clustering, classification and forecasting methods to forecast the 
evolution of multivariate time series. This approach groups windows of the signal with similar 
features into different clusters and applies a specific univariate forecasting method to each 
group, which is trained only with the corresponding windows. 

It does not applies a theoretical "ensemble" methodology, 
but the combination of clustering and classification methods to improve the performance 
of the univariate forecasting
algorithms, since they are trained only with similar windows of the series. The aim is that 
the forecasting methods will be able to learn more specific patterns, so the accurate of the estimates
will be improved.

----------
Installing
----------

This algorithm depends upon ``scikit-learn``, and thus ``scikit-learn``'s dependencies
such as ``numpy`` and ``scipy``. It also requires the specific libraries for forecasting method, 
such as ``statsmodels`` or ``keras``, or plotting, ``matplotlib``.

An ``environment.yml`` file is provided to install dependencies through conda.

.. code:: bash

    conda env create -f environment.yml
    
Otherwise, it is possible to use different versions of the libraries, but maybe some adjustments of the 
windowed methods are required.

---------------
How to use EnsembleModel
---------------

Two predefined example scripts are provided for univariate and multivariate time series. The developed algorithm 
is used as:

.. code:: python

    from model.ensemble_model import EnsembleModel    
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans
    from regresion_models.arima_model import ArimaModel

    # Cluster class
    kmeans = KMeans(n_clusters=7, random_state=42)
    
    # Classifier
    clf = GaussianNB()

    # Forecasting models
    pred_models = [ArimaModel((5, 0, 1)]
    
    # Set ensemble model params
    ensemble_model = EnsembleModel(kmeans, pred_models, classifier_model=clf)
    
    # Train model
    ensemble_model = ensemble_model.fit(X_sc_train, y_sc_train)
    
    # Forecast
    forecast_scaled = ensemble_model.predict(X_sc_test, H)

Before, time series have to be windowed using the specific parameters:

    from window_functions.windowing_data import WindowingData
    
    # Data to window matrix
    X_train, y_train = WindowingData.window_data_predictions(train_df.to_numpy(), W, H, S)  # X(N, W), y(N, H)
    X_test, y_test = WindowingData.window_data_predictions(test_df.to_numpy(), W, H, S)  # X(N, W), y(N, H)

Moreover, forecasting classical methods must be adapted in order to support a set of windowed signals. Some
example are provided in regresion_models directory, but also it is also possible to develop new ones.
It is recommended to follow the predefined structure, fit and predict methods are mandatory. Besides, it must be
verified that different instances of the class are created for every window cluster. Thus, it is also recommended
to define a function which generates an specific instance for each group according to the 
configuration parameters introduced to the model class.
