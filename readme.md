# Credit Scoring Model (CIBIL Score Prediction)

This project is a **statistical credit scoring model** that predicts a customer's likelihood of defaulting on a loan and generates a **CIBIL-like score (300-900)**. The model uses **Logistic Regression** with features such as **credit utilization, on-time payments, and past defaults** to assess creditworthiness.

## 📌 Features
- **Preprocesses financial data** (categorical & numerical features).
- **Handles class imbalance** using **SMOTE**.
- **Trains a Logistic Regression model** to predict credit risk.
- **Calculates feature importance** and contribution percentages.
- **Normalizes model output** to a **CIBIL score (300-900)**.
- **Visualizes model performance** with an **ROC curve and credit score distribution**.

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/credit-scoring-model.git
cd credit-scoring-model
```

### 2️⃣ Install Dependencies
You need Python **3.8+** and the required libraries. Install them using:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Model
```bash
python credit_score.py
```

---

## 📦 Requirements
The model depends on the following Python libraries:

```txt
numpy
pandas
scikit-learn
imblearn
matplotlib
seaborn
```

You can install them manually with:
```bash
pip install numpy pandas scikit-learn imblearn matplotlib seaborn
```

---

## 📊 Implementation Steps

### **1️⃣ Data Preprocessing**
- Drops unnecessary columns (e.g., `CustomerID`).
- Encodes categorical variables.
- Scales numerical data using `StandardScaler`.

### **2️⃣ Handling Class Imbalance**
- Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to balance defaulters and non-defaulters.

### **3️⃣ Train Logistic Regression Model**
- Splits data into training/testing sets.
- Trains a **Logistic Regression** model.
- Evaluates with **classification metrics (AUC-ROC, confusion matrix, etc.)**.

### **4️⃣ Feature Importance & Contribution**
- Extracts **feature importance** from model coefficients.
- Calculates **percentage contribution** of each factor to credit score.

### **5️⃣ Credit Score Normalization**
- Converts predicted probabilities into **CIBIL score range (300-900)**.

### **6️⃣ Model Performance Visualization**
- **ROC Curve** to analyze classification performance.
- **Histogram** to visualize predicted CIBIL scores.

---

## 📈 Example Outputs

### Feature Contribution (%)
| Feature               | Contribution (%) |
|-----------------------|----------------|
| Credit Utilization    | 35.2%          |
| On-Time Payments     | 28.4%          |
| Past Defaults        | 15.8%          |
| Total Debt           | 10.6%          |

### ROC Curve
AUC Score Example: **0.85** _(Higher is better)_

---

## 🛠 Future Improvements
- Using **Actual CIBIL Dataset** for better accuracy.
- Try advanced models like **Random Forest, XGBoost**.
- Deploy as an **API or Web App**.
- Optimize **feature selection** for better accuracy.

---

## 📜 License
This project is **MIT Licensed**.

---

## 🙌 Contributing
Feel free to contribute by creating **issues** or **pull requests**.

---

## 📩 Contact
For questions, reach out via **yuvan7480@gmail.com** or create a GitHub issue.

---