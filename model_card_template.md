 # Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Jarom Southworth created this model as part of the Udacity Machine Learning DevOps course assessment, in September 2026. It is a RandomForestClassifier from scikit-learn version 1.5.1 running on Python 3.10. All hyperparameters are scikit-learn defaults (100 trees, no depth limit) except `random_state=42`, which makes training reproducible. The model predicts whether a person's annual income is more than $50,000 or at most $50,000. Six numeric features pass through unchanged, and the eight categorical features are one-hot encoded into 102 columns, so the model receives 108 input features. The income label is binarized, with 1 meaning more than $50,000. No hyperparameter tuning or cross-validation was performed.

## Intended Use
The model is a demonstration of deploying a machine learning pipeline, intended for students and developers learning ML deployment to production. It is not suitable for estimating poverty or extreme wealth, and it must not be used for decisions about an individual's credit, employment, housing, or insurance. It sorts people into two broad buckets anchored at $50,000, and that threshold reflects 1994 dollars.

## Training Data
The data is the UCI Census Income (Adult) dataset, extracted from the 1994 US Census, with 32,561 rows. The features are age, workclass, fnlgt (a census sampling weight, used here as an ordinary numeric feature), education, education-num (a numeric version of education), marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. The label is whether annual income is more than $50,000. About 76% of rows are labeled at most $50,000, so the classes are imbalanced. Unknown values appear as the text "?" in workclass, occupation, and native-country, and the model treats "?" as its own category. A random 80% of the rows (26,048) were used for training.

## Evaluation Data
The remaining random 20% of the rows (6,513) were held out as the test set using a fixed random seed. The test data was not used for training. It was processed with the encoder and label binarizer fitted on the training data only, so no information from the test set influenced preprocessing.

## Metrics
The model is scored with precision, recall, and F1. Precision is the share of people predicted to earn more than $50,000 who really do, and the model scores 0.7419. Recall is the share of people who really earn more than $50,000 whom the model identifies, and the model scores 0.6384, meaning it misses about a third of high earners. F1 is the harmonic mean of the two and scores 0.6863. These metrics were chosen over accuracy because a model that always predicts "at most $50,000" would be correct about 76% of the time while finding no high earners at all.

Performance on data slices, one for every value of every categorical feature, is saved in `slice_output.txt`. The scores vary widely across groups:
- By sex, recall is 0.515 for women (2,126 test rows) and 0.660 for men (4,387 rows), with F1 of 0.602 and 0.700.
- By race, F1 is 0.685 for White (5,595 rows) and 0.667 for Black (599 rows). Scores for Amer-Indian-Eskimo (71 rows) and Other (55 rows) are too uncertain to interpret.
- Recall is about 0.69 for married people but 0.37 for divorced people and 0.43 for never-married people.
- Slices with very few rows give unreliable scores. For example, Cambodia has 3 test rows and scores 1.0 on every metric.

## Ethical Considerations
The model uses sensitive characteristics, including age, sex, race, and native-country. Dropping them would not remove the problem, because marital-status, relationship, occupation, and native-country correlate with them and act as proxies. The data also records income patterns from 1994, including pay disparities of that era, and a model trained on it can learn and repeat them. The slice results show one effect: the model finds high-earning women less often than high-earning men, with recall of 0.515 against 0.660. The categories are also those of the 1994 census, and sex is recorded as binary, so the model says nothing about people outside these categories. Anyone using it should evaluate performance across these groups first.

## Caveats and Recommendations
The data is from the 1994 US Census, so insights are likely outdated, and comparisons with modern incomes need inflation adjustment. Because about 76% of people earn at most $50,000, accuracy alone would flatter the model, which is why precision, recall, and F1 are reported. The model is notably weaker for people with an unknown workclass (recall 0.40 against 0.64 overall, 389 test rows). Scores for small slices are unstable, and the metric code returns 1.0 when a score is undefined, so a perfect score on a tiny group means little. I recommend retraining on current data, tuning and cross-validating the model, and reviewing fairness across sex, race, and marital status before any use beyond learning.