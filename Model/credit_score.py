# install libs
import pandas as pd
import numpy as np
import random
from faker import Faker
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix,roc_curve
from imblearn.over_sampling import SMOTE

# syntectic data genrato
fake = Faker()
num_records = 2000

data = {
    'CustomerID': [fake.uuid4() for _ in range(num_records)], # Random chars (unknown & just Ignore)
    'Age': [random.randint(18, 75) for _ in range(num_records)],
    'Income': [random.randint(20000, 150000) for _ in range(num_records)],
    'CreditLimit': [random.randint(5000, 50000) for _ in range(num_records)],
    'CreditUtilization': [random.uniform(0, 1) for _ in range(num_records)],
    'LatePayments': [random.randint(0, 10) for _ in range(num_records)],
    'LoanAmount': [random.randint(0, 100000) for _ in range(num_records)],
    'LoanTenure': [random.randint(1, 60) for _ in range(num_records)],
    'CreditAge': [random.randint(1, 300) for _ in range(num_records)], # in months
    'NumCreditLines': [random.randint(1, 10) for _ in range(num_records)],
    'CreditInquiries': [random.randint(0, 5) for _ in range(num_records)],
    'Defaulted': [random.choice([0, 1]) for _ in range(num_records)] # 0: No default, 1: Default
}

df = pd.DataFrame(data)

df

df['LatePaymentRatio'] = df['LatePayments'] / (df['LoanTenure'] + 1)  # +1 to avoid division by zero
df['DebtToIncomeRatio'] = df['LoanAmount'] / df['Income']
df['AverageCreditAge'] = df['CreditAge'] / (df['NumCreditLines'] + 1)  # +1 to avoid zero division

# Binning Credit Age to categories
bins = [0, 60, 120, 180, 240, 300]
labels = ['0-5 years', '5-10 years', '10-15 years', '15-20 years', '20-25 years']
df['Credit Age Category'] = pd.cut(df['CreditAge'], bins=bins, labels=labels, right=False)

# if req (optional) , only for better understandings of terms
df = df.rename(columns={
    'LatePayments': 'Late_Payments_Count',
    'CreditUtilization': 'Credit_Utilization_Ratio',
    'DebtToIncomeRatio': 'Debt-to-Income Ratio',
    'NumCreditLines': 'Number of Credit Lines'
})

df.head()

# preprocessing

# remove not actual need columns
df_model = df.drop(columns=['CustomerID'])

# define numerical vals
categorical_features = ['Credit Age Category']
numerical_features = [col for col in df_model.columns if col not in categorical_features + ['Defaulted']]

# pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

X = df_model.drop(columns=['Defaulted'])
y = df_model['Defaulted']
X_transformed = preprocessor.fit_transform(X)

print(f"Transformed feature shape: {X_transformed.shape}")

# apply smoke
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_transformed, y)
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Training
# 80% train - 20% test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nAUC-ROC Score:", roc_auc_score(y_test, y_prob))
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

cibil_min, cibil_max = 300, 900
y_test_cibil = cibil_min + (y_prob - y_prob.min()) * (cibil_max - cibil_min) / (y_prob.max() - y_prob.min())

# Display first few predicted scores
print("Sample CIBIL Scores:")
print(y_test_cibil[:10])

# Calculate feature importance using the coefficients of the Logistic Regression model
feature_importance = pd.DataFrame({
	'Feature': numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out()),
	'Importance': model.coef_[0]
})

# Calculate absolute importance and contribution percentage
feature_importance['Absolute_Importance'] = feature_importance['Importance'].abs()
feature_importance['Contribution (%)'] = (feature_importance['Absolute_Importance'] / feature_importance['Absolute_Importance'].sum()) * 100

# Display the contribution of top 10 factors
print("Top 10 Factors Contributing to the Credit Score:")
print(feature_importance[['Feature', 'Contribution (%)']].sort_values(by='Contribution (%)', ascending=False).head(10))

## Plot contribution of factors (Fixed Warning)
plt.figure(figsize=(10, 5))
sns.barplot(
    x='Contribution (%)',
    y='Feature',
    hue='Feature',  # Assign `y` variable to `hue`
    data=feature_importance.sort_values(by='Contribution (%)', ascending=False).head(10),
    palette='coolwarm',
    legend=False  # Hide legend
)
plt.xlabel('Contribution (%)')
plt.ylabel('Feature')
plt.title('Contribution of Factors to Credit Score')
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# score distribution
plt.figure(figsize=(8, 5))
sns.histplot(y_test_cibil, bins=30, kde=True, color='green')
plt.xlabel('CIBIL Score')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Credit Scores')
plt.show()
