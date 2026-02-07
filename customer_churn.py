#importing the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle

#data loading and understanding
df=pd.read_csv(r"C:\Users\preks\OneDrive\Desktop\customer_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
print(df.head())
pd.set_option("display.max_columns",None)
print(df.head(2))
print(df.info()) 

#dropping customer id column as it is not required for the model
df=df.drop(columns=["customerID"])
print(df.head(2))

print(df.columns)
#printing the unique values 

numerical_feature_list=["TotalCharges","MonthlyCharges","tenure"]
for col in df.columns:
    if col not in numerical_feature_list:
        print(col,":",df[col].unique())
        print("-"*50)

print(df.isnull().sum())

print(df[df["TotalCharges"]==" "])
print("\nnumber of rows where total charges = " " ")
print(len(df[df["TotalCharges"]==" "]))

df["TotalCharges"]=df["TotalCharges"].replace(" ","0.0")
df["TotalCharges"]=df["TotalCharges"].astype(float)

print(df.info())

#checking the class distribution of target column(churn)
print(df["Churn"].value_counts())

#insights:
#1.removed customer id as it is not required for modelling
#2.No missing values in the dataset
#3.missing values in the total charges column were replaced with 0
#4.class imbalanced identified in the target column

#exploratory data analysis(EDA)
print(df.shape)
print(df.columns)
print(df.describe())

#numerical feature analysis
#understand the distribution of the numerical features
def plot_histogram(df,column_name):
    plt.figure(figsize=(5,3))
    sns.histplot(df[column_name],kde=True)
    plt.title(f"distribution of {column_name}")

    #calculate the mean and median values for the columns
    col_mean=df[column_name].mean()
    col_median=df[column_name].median()

    #add vertical lines for mean and median
    plt.axvline(col_mean,color="red",linestyle="--",label="mean")
    plt.axvline(col_median,color="green",linestyle="-",label="median")

    plt.legend()

    plt.show()


plot_histogram(df,"tenure")
plot_histogram(df,"MonthlyCharges")
plot_histogram(df,"TotalCharges") #right skewed

#box plot for numerical features(to see outliers)
def plot_boxplot(df,column_name):
    plt.figure(figsize=(5,3))
    sns.boxplot(y=df[column_name])
    plt.title(f"distribution of {column_name}")
    plt.ylabel(column_name)
    plt.show()

plot_boxplot(df,"tenure")
plot_boxplot(df,"MonthlyCharges")
plot_boxplot(df,"TotalCharges") 
#there are no outliers

#correlation heatmap for numerical columns

#correlation matrix- heatmap
plt.figure(figsize=(8,4))
sns.heatmap(df[["tenure","MonthlyCharges","TotalCharges"]].corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("correlation heatmap")
plt.show()

#categorical features - analysis

print(df.columns)
print(df.info())
#count plot for categorical features

categorical_cols=df.select_dtypes(include="object").columns.to_list()
print(categorical_cols)
categorical_cols=["SeniorCitizen"]+ categorical_cols
print(categorical_cols)
for col in categorical_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f"Count plot of {col}")
    plt.show()

#there is a imbalance in the target column
#data preprocessing

print(df.head())


#Label encoding of target column
df["Churn"]= df["Churn"].replace({"Yes":1,"No":0})
print(df.head(3))
print(df["Churn"].value_counts())

#Label encoding of categorical features
#features with object data type(senior citizen dtype=int64)
object_cols=df.select_dtypes(include="object").columns
print(object_cols)

#initialize a dictionary to save the encoders
encoders={}
#apply label encoding and store the encoders
for column in object_cols:
    label_encoder=LabelEncoder()
    df[column]=label_encoder.fit_transform(df[column])
    encoders[column]=label_encoder

#save the encoders to a pickle file
with open("encoders.pkl","wb") as f:
    pickle.dump(encoders,f)

print(encoders)

print(df.head())

#Training and test data 

#splitting the features and target
x=df.drop(columns=["Churn"])
y=df["Churn"]

#split training and test data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) #20% is the test data and train data is 80%
print(y_train.shape)
print(y_train.value_counts()) 

#synthetic minority over sampling technique(SMOTE)
smote=SMOTE(random_state=42)
X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)
print(y_train_smote.shape)
print(y_train_smote.value_counts()) 

#5.model training

#training with default hyperparameters
#dictionary of models
models={
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

#dictionary to store the cross validation results
cv_scores={}

#perform 5 fold cross validation for each model
for model_name,model in models.items():
    print(f"Training {model_name} with default parameters")
    scores=cross_val_score(model,X_train_smote,y_train_smote,cv=5,scoring="accuracy")
    cv_scores[model_name]=scores
    print(f"{model_name} cross-validation accuracy:{np.mean(scores):.2f}")
    print("-"*50)

print(cv_scores)

#random forest has high accuracy - 0.84 compared to other models with efault parameters
rfc=RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote,y_train_smote)
print(y_test.value_counts())

#6. model evaluation
#evaluate on test data
y_test_pred=rfc.predict(X_test)
print("accuracy score:\n",accuracy_score(y_test,y_test_pred))
print("confusion matrix:\n",confusion_matrix(y_test,y_test_pred))
print("Classification report:\n",classification_report(y_test,y_test_pred))

#save the train model as a pickel file
model_data={"model":rfc,"features_names":x.columns.tolist()} #x is the all the features except churn (we drop churn)
with open("customer_churn_model.pkl","wb") as f:
    pickle.dump(model_data,f)

#7. load the saved model and build a predictive system
#load the saved model and feaures name
with open("customer_churn_model.pkl","rb") as f:
    model_data=pickle.load(f)
    loaded_model=model_data["model"]
    features_name=model_data["features_names"]

print(loaded_model)
print(features_name)
customer_data_input = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

input_data_df=pd.DataFrame([customer_data_input])
with open("encoders.pkl","rb") as f:
    encoders=pickle.load(f)

print(input_data_df.head())
#encode categorical features using the saved encoders
for column, encoder in encoders.items():
    input_data_df[column]=encoder.transform(input_data_df[column])

#make a prediction
prediction=loaded_model.predict(input_data_df)
pred_prob=loaded_model.predict_proba(input_data_df)
print(prediction)
print(f"Prediction:{'churn' if prediction[0]==1 else 'No churn'}")
print(f"prediction probability:{pred_prob}")
#prediction probability:[[0.83 0.17]] 0.83-0.83 → 83% probability of No Churn (class 0)
#0.17 → 17% probability of Churn (class 1)
