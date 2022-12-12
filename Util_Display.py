import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay


class Util_display:
    """
    Class to implement a tool display based on sklearn module metrics and matplotlib.

    Parameters:
    - model -- model to display.
    """
    def __init__(self, model) -> None:
        self.model = model
        self.results = model.hyper_search.cv_results_

    def hyper_results(self):
        """
        reseach the hyper parameters

        Returns a dataframe for parameters
        """
        df = pd.DataFrame.from_dict(self.model.hyper_search.cv_results_, orient='index').rename_axis(
            'Splits results', axis=0).rename_axis('Split number', axis=1)
        return df

    def best_estimators(self, n_top=3):
        """
        reseach the best best_estimator

        """
        scorer = "Accuracy"
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(self.results["rank_test_{}".format(scorer)] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print(
                    "Mean validation {0}: {1:.3f} (std: {2:.3f})".format(
                        scorer,
                        self.results["mean_test_{}".format(scorer)][candidate],
                        self.results["std_test_{}".format(scorer)][candidate],
                    )
                )
                print("Parameters: {0}".format(self.results["params"][candidate]))
                print("")

    def class_report(self, y_true, y_pred, mean_only=True):
        """
        generate the report for plot later

        return classfication report (dataframe)

        """
        report = classification_report(y_true, y_pred, target_names=self.model.class_names, zero_division=0, output_dict=True)
        report = pd.DataFrame.from_dict(report, orient='index')
        if mean_only:
            report = report.iloc[-4]
        return report

    def plot(self, param_abscissa):
        """
        Inspired from https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html 

        plot the graph of results
        """
        plt.figure(figsize=(13, 8))

        plt.title("GridSearchCV evaluating for {}".format(self.model.__class__.__name__), fontsize=16)

        plt.xlabel(param_abscissa)
        plt.ylabel("Score")
        x_size = max(self.results["param_{}".format(param_abscissa)].data)
        ax = plt.gca()
        ax.set_xlim(0, x_size)
        ax.set_ylim(0, 1.01)

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(self.results["param_{}".format(param_abscissa)].data, dtype=float)

        for scorer, color in zip(sorted(self.model.scorers), ["g", "k", "r", "b"]):
            for sample, style in (("train", "--"), ("test", "-")):
                sample_score_mean = self.results["mean_%s_%s" % (sample, scorer)]
                sample_score_std = self.results["std_%s_%s" % (sample, scorer)]
                ax.fill_between(
                    X_axis,
                    sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == "test" else 0,
                    color=color,
                )
                ax.plot(
                    X_axis,
                    sample_score_mean,
                    style,
                    color=color,
                    alpha=1 if sample == "test" else 0.7,
                    label="%s (%s)" % (scorer, sample),
                )

            best_index = np.nonzero(self.results["rank_test_%s" % scorer] == 1)[0][0]
            best_score = self.results["mean_test_%s" % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot(
                [
                    X_axis[best_index],
                ]
                * 2,
                [0, best_score],
                linestyle="-.",
                color=color,
                marker="x",
                markeredgewidth=3,
                ms=8,
            )

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)
        plt.savefig("./graph/{}.png".format(self.model.__class__.__name__), transparent=False)
        plt.show()

