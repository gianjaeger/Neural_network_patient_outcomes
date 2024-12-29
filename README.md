# Neural_network_patient_outcomes
This project was completed as part of a take-home exam. The aim is to develop a neural network for binary classification of patient outcomes and compare its performance with a logistic regression and XGBoost.

The project includes code for the following steps: 
1. Data processing (a. delete columns with more than 50% of values missing and b. iteratively impute missing datapoints with a random forest regressor to accomodate non-linear relationships and different variable types).
2. Conduct random search to define the most effective neural network architecture.
3. train a neural network with the parameters defined under random search and run it on the validation data.
4. Test the neural network and compare its performance with that of a logistic regression and XGBoost.
5. Conduct PCA to derive the number of "meaningful components" to the analysis.
