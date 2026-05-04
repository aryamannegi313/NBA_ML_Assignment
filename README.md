# Predicting NBA Game Outcomes from Player Performance Statistics Using Supervised Machine Learning

## Project Overview
This supervised learning project predicts whether an NBA player's team won or lost a game using player-level box score statistics from the NBA Daily Leaders 2024-25 dataset.

## Dataset
- Dataset name: NBA - Player Stats - Season 24/25
- Dataset file used: `nba_dailyleaders_full_24_25.csv`
- Dataset source: https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425
- Rows: 28,265
- Columns: 27

Each row represents one player performance in one NBA game.

## Target Variable
The target variable is `Result`.

The values are converted as follows:
- `W` = 1 = Win
- `L` = 0 = Loss

This is a binary classification problem.

## Research Questions
1. How effectively can baseline supervised learning models predict NBA game outcomes using individual player performance statistics?
2. Which supervised learning model achieves the best predictive performance for NBA game outcome prediction?
3. How do preprocessing strategies affect model performance?
4. Which player performance statistics contribute most strongly to predicting wins and losses?
5. Does the ranking of models change when different evaluation metrics are used?
6. How robust is the selected model under different validation settings and data perturbations?
7. Which model provides the best balance between performance, interpretability, robustness, and practical usefulness?

## Machine Learning Models
The notebook trains and compares:
- Logistic Regression
- Decision Tree Classifier
- k-Nearest Neighbors
- Random Forest Classifier
- Gradient Boosting Classifier

## Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Confusion matrix

## Important Methodological Note
The main model excludes the `+/-` column because plus/minus is highly related to the final game result and may cause outcome leakage. This makes the prediction problem more realistic.

## How to Run the Notebook
1. Open Anaconda Navigator.
2. Launch Jupyter Notebook.
3. Open the project folder.
4. Open `NBA_ML_Assignment.ipynb`.
5. Click `Kernel > Restart & Run All`.

The notebook will automatically create/update the `tables/` and `figures/` folders.

## Repository Structure
```
NBA_ML_Submission/
|-- NBA_ML_Proposal.pdf
|-- NBA_ML_Assignment.ipynb
|-- nba_dailyleaders_full_24_25.csv
|-- README.md
|-- requirements.txt
|-- Dataset_Link.txt
|-- tables/
|-- figures/
```

## Outputs
The notebook saves all result tables as CSV files in the `tables/` folder and all figures as PDF and PNG files in the `figures/` folder.

## Final Recommendation
The final model is selected based on predictive performance, interpretability, robustness, and practical usefulness.
