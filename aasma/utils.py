import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def z_table(confidence):
    """Hand-coded Z-Table

    Parameters
    ----------
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The z-value for the confidence level given.
    """
    return {
        0.99: 2.576,
        0.95: 1.96,
        0.90: 1.645
    }[confidence]


def confidence_interval(mean, n, confidence):
    """Computes the confidence interval of a sample.

    Parameters
    ----------
    mean: float
        The mean of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The confidence interval.
    """
    return z_table(confidence) * (mean / math.sqrt(n))


def standard_error(std_dev, n, confidence):
    """Computes the standard error of a sample.

    Parameters
    ----------
    std_dev: float
        The standard deviation of the sample
    n: int
        The size of the sample
    confidence: float
        The confidence level for the z-value.

    Returns
    -------
        The standard error.
    """
    return z_table(confidence) * (std_dev / math.sqrt(n))


def plot_confidence_bar(names, means, std_devs, N, title, x_label, y_label, confidence, show=False, filename=None, colors=None, yscale=None):
    """Creates a bar plot for comparing different agents/teams.

    Parameters
    ----------

    names: Sequence[str]
        A sequence of names (representing either the agent names or the team names)
    means: Sequence[float]
        A sequence of means (one mean for each name)
    std_devs: Sequence[float]
        A sequence of standard deviations (one for each name)
    N: Sequence[int]
        A sequence of sample sizes (one for each name)
    title: str
        The title of the plot
    x_label: str
        The label for the x-axis (e.g. "Agents" or "Teams")
    y_label: str
        The label for the y-axis
    confidence: float
        The confidence level for the confidence interval
    show: bool
        Whether to show the plot
    filename: str
        If given, saves the plot to a file
    colors: Optional[Sequence[str]]
        A sequence of colors (one for each name)
    yscale: str
        The scale for the y-axis (default: linear)
    """

    errors = [standard_error(std_devs[i], N[i], confidence) for i in range(len(means))]
    fig, ax = plt.subplots()
    # plt.rcParams.update({'figure.figsize':(3*len(names), 4), 'font.size': 22, 'font.weight' : 'bold'})
    x_pos = np.arange(len(names))
    ax.bar(x_pos, means, yerr=errors, align='center', alpha=0.5, color=colors if colors is not None else "gray", ecolor='black', capsize=10)
    ax.set_ylabel(y_label, fontdict={ 'size': 13, 'weight' : 'bold' })
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.set_xlabel(x_label, fontdict={ 'size': 13, 'weight' : 'bold' })
    ax.yaxis.grid(True)
    if yscale is not None:
        plt.yscale(yscale)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


def compare_results(results, confidence=0.95, title="Agents Comparison", metric="Response-rate", colors=None, filename=None):

    """Displays a bar plot comparing the performance of different agents/teams.

        Parameters
        ----------

        results: dict[str, dict[str, np.ndarray]]
            A dictionary where keys are the names and the values sequences of trials
        confidence: float
            The confidence level for the confidence interval
        title: str
            The title of the plot
        metric: str
            The name of the metric for comparison
        colors: Sequence[str]
            A sequence of colors (one for each agent/team)

        """

    results = results[metric]
    teams = list(results.keys())
    if isinstance(list(results.values())[0], np.ndarray):

        means = [result.mean() for result in results.values()]
        stds = [result.std() for result in results.values()]
        N = [result.size for result in results.values()]

        plot_confidence_bar(
            names=teams,
            means=means,
            std_devs=stds,
            N=N,
            title=title,
            x_label="", y_label=f"Avg. {metric}",
            confidence=confidence, show=True, colors=colors,
            filename=filename
        )
    else:
        import pandas as pd

        agents = [agent for agent in list(results.values())[0].keys()]

        agents_data = { agent: [] for agent in agents }

        for agent in agents:
            for team in teams:
                agents_data[agent].append(results[team][agent].mean())

        # create a dataframe
        df = pd.DataFrame(agents_data, index=teams)
        ax = df.plot.bar(rot=0, figsize=(1.25 * len(teams), 4), title=title)
        ax.legend(ncol=len(agents))
        ax.set_xlabel("")
        ax.set_ylabel(f"Avg. {metric}")
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        plt.show()
        plt.close()

def print_results(results):
    teams = [team for team in list(results.values())[0]]
    metrics = list(results.keys())

    for team in teams:
        print("%s Team:" %team)
        for metric in metrics:
            metric_results = results[metric][team]
            if isinstance(metric_results, dict):
                print("\t%s:" %metric)
                for agent in metric_results.keys():
                    print("\t\t%s: %.3f" %(agent, metric_results[agent].mean()))
            else: # ndarray
                print("\t%s: %.3f" %(metric, metric_results.mean()))
