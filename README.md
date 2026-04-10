# 📊 Customer Churn Prediction

## 📌 Project Description
This project focuses on predicting customer churn in a telecommunications company using machine learning techniques.

The goal is to determine the probability that a customer will stop using the service.

---

## 📁 Project Structure

churn_project/  
│  
├── data/ # dataset  
├── models/ # saved models  
├── src/  
│ ├── train.py # model training  
│ ├── evaluate.py # model evaluation  
│  
├── notebooks/  
│ └── eda.ipynb # exploratory data analysis  
│  
├── app.py # Streamlit application  
├── requirements.txt  
├── Dockerfile  
├── README.md  
└── .gitignore  

---

## 📊 Exploratory Data Analysis (EDA)

The following steps were performed:
- distribution analysis  
- missing values check  
- correlation analysis  

---

## 🤖 Models

The following models were used:
- Logistic Regression  
- Random Forest  

### 📈 Results:

| Model               | Accuracy | F1-score |
|--------------------|----------|----------|
| Logistic Regression | 0.82     | 0.84     |
| Random Forest       | 0.94     | 0.95     |

✅ Random Forest achieved the best performance

---

## 📉 Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

---

## 🌐 Web Application

Implemented using Streamlit.

Features:
- user input for customer data  
- churn prediction  
- probability visualization  

---

## ▶️ How to Run

### 1. Install dependencies

pip install -r requirements.txt


### 2. Train the model

python src/train.py


### 3. Evaluate the model

python src/evaluate.py


### 4. Run the web app

streamlit run app.py


---

## 🐳 Docker

### Build image:

docker build -t churn-app .


### Run container:

docker run -p 8501:8501 churn-app


---

## 📌 Conclusion

A customer churn prediction model was successfully developed.

Random Forest showed the best performance and was used in the final application.

The project includes a full pipeline:
- data preprocessing
- model training
- evaluation
- integration into a web application
