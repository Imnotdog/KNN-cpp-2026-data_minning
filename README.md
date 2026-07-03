# KNN and Ensemble KNN in C++

This is a C++ implementation of the K-Nearest Neighbors (KNN) algorithm and an Ensemble KNN classifier. The project includes datasets for testing.

This repository was documented with the help of Gemini.

## Features

- **Support for different data types**: You can use both categorical and numerical variables.
  - `NUMERICAL`: Continuous numbers (automatically normalized using Z-score).
  - `CATEGORICAL`: Categories (supports One-Hot Encoding or Label Encoding).
  - `BOOL`: True/False values.
  - `IGNORE`: Columns that are not used (like IDs or names).
- **Z-Score Normalization**: Scales numerical data automatically.
- **Ensemble KNN**: Combines multiple KNN models using bootstrapping (Bagging) to improve accuracy.
- **CSV Parser**: Reads data from CSV files and prepares them for the model.
- **Data Shuffling**: Shuffles and splits data into training and test sets.

## File Structure

- `KNN.h` / `KNN.cpp`: The core KNN algorithm and distance calculations.
- `EnsembleKNN.h` / `EnsembleKNN.cpp`: The ensemble model code.
- `Data.h` / `Data.cpp`: Handles reading CSV files and preprocessing data.
- `Feature.h` / `Feature.cpp`: Defines data types and handles encoding.
- `shuffleData.h` / `shuffleData.cpp`: Splits data into training and testing sets.
- `main.cpp`: The main entry point to run experiments.
- `makefile`: Compilation commands for Windows.

## Datasets and Sources

The repository contains datasets for testing. Here are the sources:

1. **Titanic Dataset** (`TitanicTraining.csv`, `TitanicTest.csv`)
   - Source: Kaggle Titanic Competition
   - Link: https://www.kaggle.com/c/titanic
   - Description: Predicts passenger survival. Contains mixed numerical and categorical data.

2. **Wine Dataset** (`WineTraining.csv`, `WineTest.csv`)
   - Source: UCI Machine Learning Repository
   - Link: https://archive.ics.uci.edu/ml/datasets/wine+quality
   - Description: Predicts wine quality using numerical features.

3. **Bank Dataset** (`bank.csv`, `bankTraining.csv`, `bankTest.csv`)
   - Source: UCI Machine Learning Repository
   - Link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
   - Description: Predicts if a customer will subscribe to a term deposit.

## How to Compile and Run

To compile and run the project, use:

```bash
make
```

To clean the compiled files:

```bash
make clean
```

## Example Code

You can define data types in `main.cpp` like this:

```cpp
#include "Data.h"
#include "KNN.h"

vector<Attrib> attribType = {
    {"PassengerId", IGNORE},
    {"Survived", BOOLLABEL},
    {"Pclass", NUMERICAL},
    {"Sex", BOOL},
    {"Embarked", CATEGORICAL}
};

Data trainingData("TitanicTraining.csv", "TitanicArttib.txt", attribType, LABEL_ENCODING);
KNN knn(5);

vector<double> accuracyLog;
unsigned int idx = 0;
knn.showAccuracy("TitanicTest.csv", trainingData, cout, accuracyLog, idx);
