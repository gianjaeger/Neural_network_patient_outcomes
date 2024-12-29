# Neural network for predicting patient outcomes
This project was completed as part of a take-home exam. The aim is to develop a neural network for binary classification of patient outcomes. The exam, titled "Introduction to Data Science and Machine Learning in Python," consisted of three parts, and this project was completed as a response to the third part. It covers content taught by Chris Russel in the class "Machine Learning."

**The project approaches the problem as follows:**

1. Delete columns with more than 50% of values missing and then iteratively impute missing data points with a random forest regressor to accommodate non-linear relationships and different variable types.
2. Conduct random search to derive the most effective neural network architecture.
3. Train a neural network with the parameters defined under random search and run it on the validation data.
4. Test the neural network and compare its performance with that of a logistic regression and XGBoost.
5. Conduct PCA to derive the number of "meaningful components" for the analysis and plot the classification errors of all three models to capture the bias vs. variance trade-off.

--------

A PDF with the proposed methodology and the final answer - as submitted to the examiners - is also included in the repository.

