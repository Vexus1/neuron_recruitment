# EEG i Rozpoznawanie Liter - Projekty klasyfikacyjne

## 📁 Struktura projektu
```
neuron_task/
│
├── eeg_authentication/               # Zadanie 2
│   ├── data/
│   │   └── autentykacja_eeg.csv     
│   ├── notebooks/
│   │   └── eeg_authentication.ipynb   
│   ├── outputs/
│   │   └── mlp_training_log.csv      
│   ├── src/
│   │   ├── models/
│   │   │   ├── mlp.py
│   │   │   └── random_forest.py
│   │   ├── data_loader.py
│   │   ├── eval.py
│   │   ├── torch_trainer.py
│   │   └── wrappers.py
│   └── main.py                        
│
├── letter_classifier/                 # Zadanie 1 
│   ├── data/
│   │   └── xyz_dataset.csv            
│   ├── notebooks/
│   │   └── model_comparison.ipynb     
│   ├── outputs/
│   │   └── cnn_training_log.csv       
│   ├── src/
│   │   ├── models/                   
│   │   │   ├── cnn.py
│   │   │   ├── knn.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── random_forest.py
│   │   │   └── svm.py
│   │   ├── data_loader.py
│   │   ├── evaluator.py
│   │   ├── torch_trainer.py
│   │   └── wrappers.py
│   └── main.py                        
|
├── requirements.txt                  
└── .gitignore                        
```

## 🧠 Projekt 1 – Klasyfikacja Liter X, Y, Z
- Celem było wytrenowanie kilku klasyfikatorów dla ręcznie pisanych liter na obrazach 28x28 (szarości).
- Porównano: Logistic Regression, KNN, Random Forest, SVM oraz CNN (PyTorch).
- Najlepiej poradził sobie CNN (~99% accuracy).
- Dane: `xyz_dataset.csv`

## ⚡ Projekt 2 – Autentykacja EEG
- Celem była klasyfikacja, czy użytkownik znał dany bodziec wizualny na podstawie fal EEG.
- Modele: Random Forest i MLP (PyTorch).
- Dane pochodziły z urządzenia MindWave Mobile 2, z kolumną `Flag` (0/1) i cechami EEG.
- Random Forest osiągnięto ~ 82% dokładności
- MLP osiągnęło ~68% dokładności – ograniczeniem była mała liczba próbek i szumność danych oraz lekkie braki wiedzy.

## 🧪 Uruchamianie
### 1. Zainstaluj środowisko
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Uruchom skrypty z folderów
```bash
.../letter_classifier/main.py

.../egg_authentication/main.py
```

### 3. Analiza i wizualizacje
W folderze `notebooks/` znajdziesz notebooki EDA z pełnym opisem i wykresami:
- `eeg_authentication_analysis.ipynb`
- `model_comprasion.ipynb`

## ✅ Wymagania
- Python 3.12+ (Korzystałem z 3.12.7 możliwe, że wcześniejsze wersje też będą poprawnie działały)
- Biblioteki z `requirements.txt`
