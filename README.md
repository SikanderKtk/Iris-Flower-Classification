🌸 Iris Flower Classification using Machine Learning
📌 Overview

This project applies machine learning techniques to classify the species of the famous Iris flower dataset (Setosa, Versicolor, Virginica).

The dataset contains 150 samples with four key features:

Sepal Length

Sepal Width

Petal Length

Petal Width

The goal is to build accurate and robust predictive models that can identify the flower species. The notebook demonstrates the end-to-end ML workflow, including EDA, preprocessing, hyperparameter tuning, model evaluation, and testing on new data.

🎯 Objectives

Perform Exploratory Data Analysis (EDA) to understand the dataset.

Train and optimize multiple machine learning models using GridSearchCV.

Compare models on the full dataset and also test predictions on completely new data.

Evaluate models using multiple metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC).

Save trained models, plots, and evaluation results as reproducible artifacts.

Strengthen understanding of classification concepts and hyperparameter tuning.

🔬 Methodology
1. Data Collection

Dataset: Iris dataset (loaded directly from sklearn.datasets).

Features: Sepal length, Sepal width, Petal length, Petal width.

Target: Iris species (Setosa, Versicolor, Virginica).

2. Exploratory Data Analysis (EDA)

Libraries: Pandas, Seaborn, Matplotlib.

Visualizations: scatter plots, pairplots, bar charts, histograms, heatmaps.

Insights:

Setosa is linearly separable.

Versicolor and Virginica overlap.

Strong correlation between petal length & width.

3. Data Preprocessing

Standardization using StandardScaler.

Training/testing split with train_test_split (stratified).

4. Model Building & Tuning

Models trained and tuned:

Logistic Regression

Support Vector Machine (SVM, RBF kernel)

Random Forest Classifier

GridSearchCV used with StratifiedKFold cross-validation for hyperparameter tuning.

5. Evaluation

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

Plots: Confusion Matrices, ROC Curves, Bar Charts of metrics.

6. Testing

Full dataset → overall validation.

New unseen inputs → custom predictions tested.

7. Saving Artifacts

Models saved with joblib.

Plots saved as PNG.

Reports exported for documentation.

📊 Findings

EDA: Clear separation of Setosa, overlap between Versicolor & Virginica.

GridSearchCV: Optimized hyperparameters improved all models.

Best Model: SVM (RBF kernel) showed the highest accuracy and F1-score.

Testing:

Full dataset → ~97% accuracy.

New inputs → Correct species predictions.

✅ Results

Best Model: SVM (RBF kernel).

Performance Summary:

Accuracy: ~97%

Weighted F1-score: Highest among all models

ROC-AUC: ~1.0

Artifacts:

Models (.joblib)

Heatmaps, Confusion Matrices, ROC Curves, EDA plots

Comparison charts

🔎 Conclusion

The Iris dataset is ideal for classification & hyperparameter tuning practice.

GridSearchCV greatly boosted model performance.

SVM with RBF kernel proved most accurate and generalizable.

Predictions on new data validated the model’s robustness.

Project demonstrates a complete ML pipeline:
EDA → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Saving → Testing.

👨‍💻 Author

Developed by Muhammad Sikander Bakht ✨
📧 Email: sikanderktk222@gmail.com

🔗 GitHub: [https://github.com/SikanderKtk]
