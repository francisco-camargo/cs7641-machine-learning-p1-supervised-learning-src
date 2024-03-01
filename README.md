# CS7641 Machine Learning - Project 1: Supervised Learning - Francisco Camargo

## Running the Code
To run `main.py` for either dataset problem, do one of the following:
`$ python main.py -c configs/config_mushroom.yaml`
`$ python main.py -c configs/config_car.yaml`

To change which machine learning algorithm is used or how they are each run, you will have to alter the yaml config files accordingly.

## LaTeX template
The LaTeX template I used for this report comes from the Vision Stanford [website](http://vision.stanford.edu/cs598_spring07/report_templates/)

## TODO
- [x] Get two datasets
  - [x] car
  - [x] monk
- [x] Preprocessing, [guide](https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9)
  - [x] maybe use less data for testing? would that help me with cancer data? nah
  - [x] test-train splitting
  - [x] startify target classes
  - [x] import .pkl files if needed
  - [x] handle [ordinal](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) data, maybe make pandas columns into [categories](https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9), maybe use [LabelEncoder](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/) on target. [This](https://datascience.stackexchange.com/questions/72343/encoding-with-ordinalencoder-how-to-give-levels-as-user-input?newreg=d2f4c1b1b5ab45f2ae31e3a08913ceb9) is what I need.
    - [x] replace df columns with fit_transform output
    - [x] set the data types to int for all
  - [x] handle [categorical](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) data
    - [x] set data type
- [ ] train
  - [ ] stop using hard-coded xlim for learning curves (right now set to 350)
  - [x] change fixed_params in yaml to only appear once
  - [x] random_state doesn't help me, what the heck... maybe it's random state of learning/validation curve? can't (seem to) control fit_time because it is measuring CPU preformance which is not constant each time I run it.
    - [x] random_state now works for GridSearchCV
  - [x] maybe move oversample param to be part of each experiment dict
  - [x] Different [scores](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) used to train
    - [x] [guide](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/) to use for imbalanced data
  - [x] [GridSearch](https://scikit-learn.org/stable/modules/grid_search.html), [guide](https://www.mygreatlearning.com/blog/gridsearchcv/)
  - [x] Over/Under sampling: [guide](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/) [guide](https://towardsdatascience.com/imbalanced-class-sizes-and-classification-models-a-cautionary-tale-part-2-cf371500d1b3), [SMOTE](https://www.kaggle.com/residentmario/oversampling-with-smote-and-adasyn)
  - [x] be able to run best version of each model
  - [ ] learning curves, [guide](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/) [sklearn guide](https://scikit-learn.org/stable/modules/learning_curve.html) [example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) [guide](https://www.dataquest.io/blog/learning-curves-machine-learning)
    - [x] % of train data on x
    - [x] iteration # on x for SVM and NN. nah
    - [ ] training time on x; what are the units? time per iteration? time per sample?
  - [x] validation/complexity curves, k-fold xvalidation  [example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py)
    - [x] use cv = ShuffleSplit? yes
  - [ ] get results ready to plot
    - [x] save output data to csv so that plotter can just read csv of training results. nah
    - [x] put dataset name in plots
    - [ ] prevent std bands from exceeding a score of 100%
- [ ] test
  - [ ] read in pkl file
  - [ ] how do I interpret results table?
- [x] ML Algos:
  - [x] Decision Tree:
    - [x] control max depth
    - [x] need to add pruning: [alpha](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)
    - [x] feature importance to see if there are irrelevant features
  - [x] add Adaboost, something about weak learners?
  - [x] NN:
    - [x] iteration .loss_curve_? or maybe use .partial_fit()? have to fit MLP directly, can't use pipeline. nah
  - [x] SVM
    - [x] iteration curve
      - [x] use validation curve code and make max_iter the independent variable
    - [x] must use at least two kernels
- [x] Report
  - [x] read Monk and cancer papers, nah
  - [x] is bias/variance a property of the training result? or the validation result? or the trained model at each point along the independent variable axis? I think it's a property of the trained model which implies it depends on the ML algo and the trianing data
  - [x] compare datasets
    - [x] summary stats
  - [x] talk about bias vs variance: [guide](https://vitalflux.com/learning-curves-explained-python-sklearn-example/) [guide](https://zahidhasan.github.io/2020/10/13/bias-variance-trade-off-and-learning-curve.html) [guide](https://www.dataquest.io/blog/learning-curves-machine-learning)
  - [x] need to tune at least two hyper params per algo
  - [x] Plots
    - [x] 2 data sets
      - [x] 2hp*5algos=10 validation curves
      - [x] 3simple+2iterative*2= 7 learning curves
    - [x] 2*(10+7)= 34
  - [x] comparing all five algorithms with tuned hyperparameters vs. total training and final test performance

## Accessing (test) performance

Hey all,

What would I do to try to get a confidence band around test/out-of-sample performance?

What if we break up a test set into several parts such that we can get a distribution of performance on purely out-of-sample data with respect to the final model we make (which uses all data in the training set, such that a cross-validation bands are an indirect indicator)

## Datasets

### Monks problem
https://www.openml.org/d/333
[UCI MONKs](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems)

* pros
  * great learning
  * continuous features
  * binary target
  * balanced target
  * *need* hidden layers
  * no missing values

* cons
  * 6 features
  * 556 samples


### Breast Cancer
https://www.kaggle.com/dhainjeamita/breast-cancer-dataset-classification
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

* pros
  * ~30 features
  * binary target
  * continuous features
  * imbalanced 1.7:1
  * can reduce dimensionality 

* cons
  * little learning
  * 569 samples
  * ok learning
  
* interesting

  * high score with few examples, but not a lot of learning with 340 training examples


## Guidance
* [cookiecutter](https://drivendata.github.io/cookiecutter-data-science/)
* setting [random seed](https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f)
* getting started with [scikit-learn](https://scikit-learn.org/stable/getting_started.html)
* cross-validation with [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)

## Unused datasets
### Synthetic Data
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

### Student Data
http://archive.ics.uci.edu/ml/datasets/Student+Performance
Prob won't use; the target is numerical index values

### Mushroom Data
http://archive.ics.uci.edu/ml/datasets/Mushroom
http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/

csv:
https://www.kaggle.com/uciml/mushroom-classification/version/1

mushroom

* pros

  * 22 features
  * 8124 samples
  * great learning

* cons

  * balanced
  * binary
  * *very* little noise
  * don't need hidden layers


### Car Valuation
http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
https://www.kaggle.com/elikplim/car-evaluation-data-set for .csv format
guide: https://www.24tutorials.com/machine-learning/case-study-decision-tree-model-for-car-quality/

car: ordinal regression/classification

* pros
  * multiclass
  * imbalanced
  * ordinal features
  * 1728 samples
  * *need* hidden layers
* cons
  * *ordinal* regression/classification
  * 6 features
  * ok learning

### Blood Transfusion
https://www.openml.org/d/1464

### Diabetes
https://www.openml.org/d/37

### NASA
https://www.openml.org/d/1067