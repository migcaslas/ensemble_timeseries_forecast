import pandas
import numpy
import matplotlib.pyplot as plt

from .series_forecasting import SeriesForecasting
from .clustering_evaluation import ClusteringEvaluation


class ModelGraphs(object):

    @classmethod
    def predicted_data_graph(cls, time_series, model, start_date, end_date, step):
        forecast_data = SeriesForecasting.forecast_time_series(time_series, model, start_date, end_date, step)
        original_data = SeriesForecasting.selected_time_period(time_series, start_date, end_date, step, model.instance_window_size)
        return cls.plot_forecast_var(original_data, forecast_data)

    @classmethod
    def plot_forecast_var(cls, original_data, forecast_data):
        if isinstance(original_data, pandas.Series):
            return cls._plot_time_series(original_data, forecast_data)
        else:
            return cls._plot_data_frame(original_data, forecast_data)

    @classmethod
    def _plot_time_series(cls, original_data_ds, forecast_data_ds):
        fig, ax = plt.subplots()
        cls._plot_subplot(ax, original_data_ds, forecast_data_ds)
        # plt.show()
        return fig

    @classmethod
    def _plot_data_frame(cls, original_data_ds, forecast_data_ds):
        n_vars = len(original_data_ds.columns)
        fig, axs = plt.subplots(n_vars, 1)
        for IDX in range(n_vars):
            column_name = original_data_ds.columns[IDX]
            cls._plot_subplot(axs[IDX], original_data_ds[column_name], forecast_data_ds[column_name])
        # plt.show()
        return fig

    @classmethod
    def _plot_subplot(cls, ax, correct_ds, forecast_ds):
        pd2plot = pandas.concat([correct_ds, forecast_ds], axis=1)
        pd2plot.columns = ["original", "predicted"]
        pd2plot.plot(ax=ax)
        ax.set_title(correct_ds.name)

    @classmethod
    def plot_time_series_labelled(cls, time_series, model, step):
        labelled_series = ClusteringEvaluation.classify_labels_time_series(time_series, model, step)
        fig, ax = plt.subplots()
        cls.plot_multicolor_line(time_series[:len(labelled_series)], labelled_series.values, ax)
        plt.show()
        return fig

    @classmethod
    def plot_multicolor_line(cls, ds, colors_labels, ax):
        unique_labels = numpy.unique(colors_labels)
        unique_labels = unique_labels[~numpy.isnan(unique_labels)]
        c_map = plt.cm.get_cmap("hsv", len(unique_labels))
        segments = cls._find_contiguous_colors(colors_labels)
        start = 0
        for seg in segments:
            end = start + len(seg)
            selected_color = [0.8, 0.8, 0.8]
            if not numpy.isnan(seg[0]):
                color_idx = numpy.where(unique_labels == seg[0])[0][0]
                selected_color = c_map(color_idx)
            ax.plot(ds.iloc[start:end].index, ds.iloc[start:end].values, lw=2, c=selected_color)
            start = end

    @staticmethod
    def _find_contiguous_colors(colors):
        # finds the continuous segments of colors and returns those segments
        segs = []
        curr_seg = []
        prev_color = ''
        for c in colors:
            if numpy.isnan(c) or c == prev_color or prev_color == '':
                curr_seg.append(c)
            else:
                segs.append(curr_seg)
                curr_seg = []
                curr_seg.append(c)
            prev_color = c
        segs.append(curr_seg)  # the final one
        return segs
