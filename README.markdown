# Obesity Level Prediction Using Machine Learning

## Project Overview
This project develops a machine learning pipeline to predict obesity levels based on lifestyle and demographic factors. The dataset includes 1900 entries with 17 features, capturing factors such as diet, physical activity, and family history. The target variable, `NObeyesdad`, categorizes individuals into seven obesity levels: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III.

The project focuses on comprehensive data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation. A custom Gaussian Naive Bayes classifier is implemented alongside other models (Random Forest, SVC, KNN, LightGBM, Decision Tree) to explore relationships between lifestyle factors and obesity. Key insights highlight the influence of diet, physical activity, and genetic factors on obesity levels, providing valuable implications for healthcare applications.

## Dataset
The dataset is divided into training (`train_dataset.csv`) and testing (`test_dataset.csv`) sets, each with 1900 rows and 17 columns. Features include:

- **Demographic Features**: Gender, Age, Height, Weight
- **Lifestyle Features**:
  - FAVC (frequent high-calorie food consumption, yes/no)
  - FCVC (frequency of vegetable consumption, scale 1-3)
  - NCP (number of main meals per day)
  - CAEC (frequency of food consumption between meals: Never, Sometimes, Frequently, Always)
  - CH2O (daily water intake, scale 1-3)
  - FAF (physical activity frequency, scale 0-3)
  - TUE (time using technology, scale 0-3)
  - CALC (alcohol consumption frequency: Never, Sometimes, Frequently, Always)
  - MTRANS (transportation mode: Automobile, Bike, Motorbike, Public Transportation, Walking)
- **Other Features**: family_history_with_overweight, SMOKE, SCC (calorie monitoring, yes/no)
- **Target Variable**: NObeyesdad (7 obesity levels)

### Dataset Characteristics
- **Missing Values**:
  - FCVC: 12 missing values imputed with mean (~2.42)
  - CALC: 28 missing values imputed with mode ("Sometimes")
- **Class Distribution**: Imbalanced, with Obesity Type III (~20%) most frequent and Insufficient Weight (~10%) least frequent.

## Installation
To run this project, ensure Python 3.6+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/obesity-prediction.git
cd obesity-prediction
pip install -r requirements.txt
```

### Requirements
Key dependencies (listed in `requirements.txt`):
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `lightgbm`

Install them using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels lightgbm
```

## Usage
1. **Prepare Data**: Place `train_dataset.csv` and `test_dataset.csv` in the project directory.
2. **Run Notebook**: Open `Obesity_Prediction_Final.ipynb` in Jupyter Notebook or JupyterLab to execute preprocessing, EDA, model training, and evaluation.
3. **Custom Naive Bayes**: The notebook includes a custom Gaussian Naive Bayes implementation using 7 features: BMI, family_history_with_overweight, Age, FAF, CH2O, NCP, CAEC.
4. **Visualizations and Evaluation**: Explore visualizations and model performance metrics (precision, recall, F1-score, confusion matrix).

Run the notebook with:
```bash
jupyter notebook Obesity_Prediction_Final.ipynb
```

## Preprocessing Steps
1. **Data Loading**: Loaded training and testing datasets using `pandas.read_csv`.
2. **Missing Value Handling**:
   - FCVC: Imputed with mean (~2.42) to preserve distribution.
   - CALC: Imputed with mode ("Sometimes") for categorical consistency.
3. **Encoding Categorical Variables**:
   - Binary (e.g., Gender, FAVC): Encoded with `LabelEncoder` (0/1).
   - Ordinal (e.g., CAEC, CALC): Mapped to numerical values (e.g., Never: 0, Always: 3).
   - Nominal (MTRANS): One-hot encoded into 5 binary columns.
   - Target (NObeyesdad): Encoded with `LabelEncoder` (0-6).
4. **Scaling Numerical Features**: Standardized with `StandardScaler` and rounded to reduce noise.
5. **Feature Engineering**: Created BMI feature using `Weight / (Height^2)`.
6. **Feature Selection**:
   - **Forward Selection**: Identified 9 key features: Weight, family_history_with_overweight, Age, CAEC, FCVC, FAF, Height, NCP, Gender.
   - **Backward Elimination**: Retained 13 features, excluding SMOKE, TUE, SCC.
   - Naive Bayes used 7 features for simplicity: BMI, family_history_with_overweight, Age, FAF, CH2O, NCP, CAEC.

## Exploratory Data Analysis (EDA)
Visualizations provided insights into data distributions and relationships:
- **Obesity Distribution Bar Plot**: Revealed class imbalance, with Obesity Type III most prevalent.
- **Target Distribution with Gender**: Showed similar patterns across genders, with slight differences (e.g., more males in Obesity Type II).
- **Gender Distribution**: Balanced (~950 males, ~950 females).
- **Categorical Variable Count Plots**: Highlighted dominance of Public Transportation (~1200) and family history of overweight (~1500).
- **Vegetable Consumption vs. Obesity (Violin Plot)**: Low FCVC (<=1) correlated with higher obesity levels; high FCVC (>2) linked to Normal/Insufficient Weight.

## Model Implementation
Six models were trained and tuned using `GridSearchCV`:
1. **Random Forest**: Tuned `n_estimators` and `max_depth` for robust ensemble predictions.
2. **SVC**: Optimized `C` and `kernel` to capture non-linear relationships.
3. **KNN**: Adjusted `n_neighbors` for local pattern recognition.
4. **LightGBM**: Tuned `n_estimators`, `learning_rate`, and `max_depth` for efficient gradient boosting.
5. **Decision Tree**: Optimized `max_depth`, `min_samples_split`, and `min_samples_leaf` for interpretability.
6. **Naive Bayes**: Custom Gaussian implementation for numerical features.

### Custom Naive Bayes
The `NaiveBayes` class is tailored for numerical features:
- **Training**: Computes class priors and feature-wise mean/variance.
- **Prediction**: Uses log-probabilities for numerical stability, selecting the class with the highest probability.
- **Features**: Uses 7 features (BMI, family_history_with_overweight, Age, FAF, CH2O, NCP, CAEC) for simplicity and interpretability.
- **Performance**: High precision for extreme classes (e.g., Obesity Type III) but lower recall for overlapping classes (e.g., Overweight Level I/II).

## Model Evaluation
Models were evaluated on a 20% test set using precision, recall, F1-score (macro-averaged), and confusion matrices:
- **Random Forest**: Balanced performance, interpretable feature importance, but computationally intensive.
- **SVC**: Effective for non-linear patterns, though slower to train.
- **KNN**: Simple but sensitive to imbalanced data and scaling.
- **LightGBM**: Robust to imbalanced data, fewer misclassifications for minority classes.
- **Decision Tree**: Highly interpretable but prone to overfitting.
- **Naive Bayes**: Efficient and precise for extreme classes, limited by feature independence assumption.

### Key Insights
- **Lifestyle Factors**: Low vegetable consumption (FCVC <= 1), frequent high-calorie food intake (FAVC, ~1600 individuals), and low physical activity (FAF) strongly correlate with higher obesity levels.
- **Genetic Influence**: Family history of overweight (~1500 individuals) is a significant predictor.
- **Transportation**: Public Transportation dominates (~1200), while Walking (~200) and Bike (~50) correlate with lower obesity levels.

## Results
- **Top Model**: LightGBM excelled in handling imbalanced data and categorical features.
- **Key Features**: Weight, family_history_with_overweight, and BMI were critical predictors.
- **Healthcare Implications**: The analysis underscores the importance of diet, physical activity, and genetic factors in obesity, informing targeted interventions.
