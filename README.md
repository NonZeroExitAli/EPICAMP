# EPIC-AMP: A Comprehensive Machine Learning Framework for Antimicrobial Peptide Classification and MIC Prediction

EPIC-AMP (Explainable Platform for Interpretable Classification of Antimicrobial Peptides) integrates machine learning, explainable AI, and web deployment to identify antimicrobial peptides (AMPs) and predict their minimum inhibitory concentration (MIC) against major pathogens.

---

## Overview

Antimicrobial peptides (AMPs) are promising alternatives to antibiotics. This project establishes a reproducible and interpretable machine learning pipeline for:

- Classifying AMPs vs non-AMPs  
- Predicting quantitative MIC values for *E. coli*, *S. aureus*, *P. aeruginosa*, and *K. pneumoniae*  
- Deploying the best-performing models as a public web tool:  
  https://nonzeroexit-epic-amp.static.hf.space/index.html

---

## Project Pipeline

![EPIC-AMP Pipeline](Classification%20(1).png)

### Steps

1. **Data Acquisition & Cleaning**  
   - Merged data from six AMP databases (APD3, CAMP, dbAMP, DRAMP, ADAM, DBAASP)  
   - Non-AMPs retrieved from UniProt  
   - CD-HIT clustering (90% identity)  
   - Sequences limited to 10–100 amino acids  
   - Class balancing using SMOTE

2. **Feature Engineering**  
   - Extracted features using ProPy3:
     - Amino Acid Composition (AAC)
     - Autocorrelation
     - Composition, Transition, Distribution (CTD)
     - Pseudo-Amino Acid Composition (PseAAC)
   - Deep embeddings from ProtBert used for regression tasks.

3. **Feature Selection**  
   - Combined techniques:
     - Variance Threshold (VT)
     - Recursive Feature Elimination (RFE)
     - Random Forest Importance
     - Boruta Algorithm  
   - Generated over 225 feature combinations for classification and regression(for each organism).

4. **Model Training**  
   - Classification Models: Random Forest (RF), Support Vector Machine (SVM), K-Nearest Neighbors (KNN)  
   - Regression Models: XGBoost, Random Forest Regressor, Support Vector Regressor, Multilayer Perceptron (MLP)  
   - Hyperparameter tuning performed using GridSearchCV.

5. **Explainable AI (XAI)**  
   - SHAP for global feature importance  
   - LIME for local, sequence-specific interpretation

6. **Deployment**  
   - Best-performing models integrated into a Streamlit-based web interface  
   - Provides live AMP classification and MIC prediction with interpretability.

---

## Key Results

| Task | Model | Performance Highlights |
|------|--------|------------------------|
| Classification | Random Forest (Boruta + all features) | Accuracy: 96.0%, F1: 95.8%, Recall: 96.0% |
| MIC Regression (E. coli, S. aureus, P. aeruginosa) | XGBoost + ProtBert | R²: 0.68–0.70 |
| MIC Regression (K. pneumoniae) | MLP + ProtBert | R²: 0.74, MSE: 0.43 |

---

## Installation

```bash
git clone https://github.com/NonZeroExitAli/EPICAMP.git
cd EPICAMP
pip install -r requirements.txt
```
## Primary Notebooks

### Classification_SourceCode.ipynb
- Loads and preprocesses AMP and non-AMP datasets  
- Extracts physicochemical features using ProPy3  
- Applies feature selection (Boruta, RFE, VT, RF importance)  
- Trains and evaluates ML classifiers (SVM, RF, KNN)  
- Computes Accuracy, F1-score, Recall, Precision, and ROC-AUC metrics  

### Regression_SourceCode.ipynb
- Preprocesses MIC data for *E. coli*, *S. aureus*, *P. aeruginosa*, and *K. pneumoniae*  
- Generates ProtBert embeddings and physicochemical descriptors  
- Trains regression models (XGBoost, RF Regressor, SVR, MLP)  
- Evaluates with R², MSE, and correlation coefficients  
- Outputs plots of true vs predicted MIC and Bland–Altman analysis  

---

## Data Format

### Classification Input

| Peptide_ID | Sequence | Activity |
|-------------|-----------|-----------|
| AMP_001 | VGGVPAGPAQ | AMP |
| NONAMP_001 | LLLKKVVGGAA | nonAMP |

### Regression Input

| Sequence | Organism | MIC (µM) |
|-----------|-----------|-----------|
| RFRPPIRRPPI | *E. coli* | 2.33 |
| GIGTKILGGVKT | *P. aeruginosa* | 7.21 |

---

## Tools and Libraries

- ProPy3 — Feature extraction  
- Imbalanced-learn (SMOTE) — Class balancing  
- SHAP & LIME — Explainable AI  
- scikit-learn, XGBoost, PyTorch — Model training  
- HuggingFace Spaces — Deployment  

---

## Contributors

- Ali Magdi — Project design, modeling, deployment  
- Ahmed Amr — Data acquisition and validation  
- Eman Badr — Supervision

