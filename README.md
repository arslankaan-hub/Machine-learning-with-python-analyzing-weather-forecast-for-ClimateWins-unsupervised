## Project Summary

This project explores how machine learning can support both short term weather classification and long term climate forecasting across Europe. Using multi decade historical weather observations from 18 European weather stations, the analysis evaluates temperature evolution, atmospheric variability, seasonal cycles and the likelihood of favourable or hazardous weather events.
Multiple models were tested, including Random Forests, Convolutional Neural Networks (CNNs), Artificial Neural Networks (ANNs), Decision Trees and prototype LSTM and GAN frameworks to identify which algorithms best classify daily conditions and which features contribute most strongly to regional climate behaviour.
The project answers key questions about classification accuracy, the detectability of climate-shift indicators, model generalisation across stations and the feasibility of generating longer-term climate scenarios aligned with ClimateWins’ 25–50-year forecasting goals.

---

## Key Questions

-Identify weather patterns outside the regional norm in Europe.
-Determine if unusual weather patterns are increasing.
-Generate possibilities for future weather conditions over the next 25 to 50 years based on current trends.
-Determine the safest places for people to live in Europe over the next 25 to 50 years.

---

## Project Structure

### **01 Project Management**  
Contains the official project brief describing ClimateWins’ objectives:

---

### **02 Data**  
(*Data files not uploaded due to size limitations*)

**Original Data**  
Raw historical CSV files from the European Climate Assessment & Dataset (ECA&D).

**Prepared Data**  
Cleaned, merged, and feature-engineered datasets

---

### **03 Scripts (Jupyter Notebooks)**  
Contains all Python notebooks used in the analysis:

- **Data Preparation & Cleaning**  
- **Gradient Descent Experiments**   
- **Decision Tree Classification**  
- **Random Forest Classification & Feature Importance**  
- **CNN for Weather Station Classification**  
- **CNN for Digit & Weather-Image Classification Experiments**  
- **Confusion Matrix Evaluation**  
 
---

### **05 Sent to Client**  
Contains the final stakeholder presentation

---

# Code Overview

The following Python libraries were used throughout the project to support data processing, exploratory analysis, classical machine learning models, deep learning architectures, hyperparameter optimisation and image based classification tasks.

---

### **Data Handling & Preprocessing**

- `pandas` – Tabular data handling  
- `numpy` – Numerical operations and reshaping  
- `seaborn` – Statistical visualisations  
- `matplotlib.pyplot` – Plotting  
- `os`, `operator`, `warnings`, `time` – Utility functions  
- `numpy.asarray` – Image loading & conversion  

---

### **Classical Machine Learning (scikit-learn)**

- `train_test_split`, `cross_val_score`, `StratifiedKFold` – Model validation  
- `LabelEncoder` – Categorical encoding  
- `accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay` – Evaluation  
- `DecisionTreeClassifier`, `plot_tree` – Tree models  
- `RandomForestClassifier` – Feature importance & event prediction  
- `GridSearchCV`, `RandomizedSearchCV` – Hyperparameter search  
- `scipy.stats.randint` – Random search distributions  

---

### **Deep Learning (TensorFlow / Keras)**

- `Sequential`, `Model` – Model construction  
- **Layers:**  
  - `Dense`, `Dropout`, `BatchNormalization`, `Flatten`, `Activation`  
  - `Conv1D`, `Conv2D`, `MaxPooling1D`, `MaxPooling2D`  
  - `LSTM` – Sequence modeling  
  - `LeakyReLU` – Alternative activation  
- **Optimisers:**  
  - `Adam`, `SGD`, `RMSprop`, `Adadelta`, `Adagrad`, `Adamax`, `Nadam`, `Ftrl`  
- **Training Utilities:**  
  - `EarlyStopping`, `ModelCheckpoint`  
  - `to_categorical` – One-hot encoding  
  - `ImageDataGenerator` – Image augmentation  

---

### **Image & Dataset Utilities**

- `keras.datasets.mnist` – Sample dataset  
- `unique`, `reshape`, `argmax` – Array tools  

---

### **Model Wrappers & Optimisation**

- `KerasClassifier` – Scikit-learn-compatible wrapper  
- `BayesianOptimization` – Automated hyperparameter tuning  

---

## Disclaimer

This dataset was sourced for educational purposes as part of the CareerFoundry Data Analytics Program.  
Historical climate data originates from the **European Climate Assessment & Dataset (ECA&D)**, publicly available for academic and research use.

All models, visualisations, and findings are intended for exploratory analysis and concept demonstration not operational climate forecasting.
