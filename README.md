🌸 Iris Flower Classification using Machine Learning
 Overview

This project applies machine learning techniques to classify the species of the famous Iris flower dataset (Setosa, Versicolor, Virginica).
The dataset contains 150 samples with four key features:

Sepal Length

Sepal Width

Petal Length

Petal Width

The goal is to build accurate and robust predictive models that can identify the flower species.
The notebook demonstrates the end-to-end ML workflow, including EDA, preprocessing, hyperparameter tuning, model evaluation, and testing on new data.

 Objectives

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

Pandas / Seaborn / Matplotlib used for visualization.

Plots created: scatter plots, pairplots, bar charts, histograms, heatmaps.

Insights:

Setosa is linearly separable.

Versicolor and Virginica show overlap.

Strong correlation between petal length & width.

3. Data Preprocessing

Standardized features using StandardScaler.

Training and testing split using train_test_split with stratified sampling.

4. Model Building & Tuning

Trained and tuned the following models:

Logistic Regression

Support Vector Machine (SVM) with RBF kernel

Random Forest Classifier

Hyperparameter Tuning:

Applied GridSearchCV to find the best parameters for each model.

Used cross-validation (StratifiedKFold) for fair performance comparison.

5. Evaluation

Metrics used: Accuracy, Precision, Recall, F1-score (macro & weighted), ROC-AUC.

Plots generated:

Confusion Matrices

ROC Curves

Bar charts of metric comparison

6. Testing on Data

Full Dataset Testing: Models tested on complete dataset for overall validation.

New Data Testing: Custom flower measurements input manually → predicted species.

7. Saving Artifacts

Trained models saved with joblib.

Visualizations saved as PNG files.

Evaluation summaries exported for documentation.

 Findings

EDA: Clear separation of Setosa, overlap between Versicolor & Virginica.

GridSearchCV Results:

Optimized hyperparameters improved all models.

SVM with RBF kernel achieved the best balance of precision and recall.

Testing:

On the full dataset → Accuracy ~97%.

On unseen new inputs → Correct predictions made in all tested cases.

 Results

 Best Model: SVM (RBF kernel, tuned via GridSearchCV).

📈 Performance Summary:

Accuracy: ~97%

Weighted F1-score: Highest among models

ROC-AUC: Close to 1.0

 Artifacts:

Models saved (.joblib)

Heatmaps, Confusion Matrices, ROC Curves, EDA plots

Comparison charts across models

 Conclusion

The Iris dataset is ideal for understanding classification and hyperparameter tuning.

GridSearchCV significantly improved model performance.

SVM with RBF kernel was the most accurate and generalizable model.

Testing on new data points confirmed the robustness of the trained classifier.

The project demonstrates a complete ML pipeline: EDA → Preprocessing → Hyperparameter Tuning → Evaluation → Saving Artifacts → Testing on New Data.

 Tech Stack

Python 3.10+

Scikit-learn

Pandas / NumPy

Seaborn / Matplotlib

Jupyter Notebook

👨‍💻 Author

Developed by [MUHAMMAD SIKANDER BAKHT] ✨
📧 Contact: [sikanderktk222@gmail.com]
🔗 GitHub: [https://github.com/SikanderKtk]
