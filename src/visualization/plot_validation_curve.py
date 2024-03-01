"""
==========================
Plotting Validation Curves
==========================

Using the following for reference:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

In this plot you can see the training scores and validation scores of an SVM
for different values of the kernel parameter gamma. For very low values of
gamma, you can see that both the training score and the validation score are
low. This is called underfitting. Medium values of gamma will result in high
values for both scores, i.e. the classifier is performing fairly well. If gamma
is too high, the classifier will overfit, which means that the training score
is good but the validation score is poor.
"""

from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_validation_curve(estimator, 
                        project_alias,
                        model_choice,
                        experiment,
                        X,
                        y,
                        param_name=None,
                        param_range=None,
                        ylim=None,
                        xscale='linear',
                        cv=None,
                        scoring=None,
                        n_jobs=None,
                        savefig=False):
    
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
        )

    train_scores_mean   = np.mean(train_scores, axis=1)
    train_scores_std    = np.std(train_scores, axis=1)
    test_scores_mean    = np.mean(test_scores, axis=1)
    test_scores_std     = np.std(test_scores, axis=1)
    
    fontsize = 11
    fontsize_ticks = fontsize - 2
    lw=2
    fig_dim_x = 4
    fig_dim_y = fig_dim_x * 0.75
    tight_pad = 0.7
    filepath = '../../report/images/' + project_alias + '/' + model_choice + '/'

    fig, ax = plt.subplots()
    ax.figure.set_size_inches(fig_dim_x, fig_dim_y)
    plt.title(project_alias + ' data, ' + model_choice + ' ' + experiment, fontdict = {'fontsize' : fontsize_ticks})
    xlabel = param_name.split('class__')[-1]
    xlabel = xlabel.split('base_estimator__')[-1]
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(scoring + ' score', fontsize=fontsize)
    if ylim is not None:
        plt.ylim(ylim)
    lw = 2
    if xscale == 'linear':
        plt.plot(param_range, train_scores_mean, label='Training',
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(param_range, test_scores_mean, label='Validation',
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    elif xscale == 'log':
        plt.semilogx(param_range, train_scores_mean, label='Training',
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label='Validation',
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    ax.tick_params(direction='in')
    fig.tight_layout(pad=tight_pad)
    plt.show()
    
    filename= project_alias + '_' + model_choice + '_' + experiment
    
    # filename= '../../report/images/' + project_alias + '/' + model_choice + '/' + experiment
    if savefig:
        plt.savefig(filepath+filename+'.png', format='png')


def main():
    plt.close('all')
    X, y = load_digits(return_X_y=True)
    estimator = SVC()
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    param_range = np.logspace(-6, -2, 10)
    plot_validation_curve(estimator,
                        project_alias='',
                        model_choice='',
                        experiment='',
                        X=X,
                        y=y,
                        param_name='gamma',
                        param_range=param_range,
                        ylim=(0.2, 1.01),
                        xscale='log',
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=-1)
    
    
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.model_selection import ShuffleSplit
    main()
