# HCMIU Data Mining Project

## Overview
This project implements various machine learning classification algorithms for analyzing Steam game data. The implementation includes multiple classifiers such as Naive Bayes, OneR, and J48 decision tree algorithms.

## Project Structure
```
.
├── src/
│   └── Implementation/
│       └── algorithms/           # Classification algorithm implementations
├── results/                      # Classification results and outputs
├── game/                         # Game-related data and resources
├── test_and_eval/               # Testing and evaluation scripts
├── review/                       # Review data and analysis
├── pre-processing/              # Data preprocessing scripts
├── production/                  # Production-ready code
├── crawl_and_clean/            # Data crawling and cleaning scripts
└── lib/                         # External libraries and dependencies
```

## Data Files
The project uses ARFF (Attribute-Relation File Format) files for data processing:
- `steam_game_data_encoded.arff`: Encoded Steam game data
- `steam_game_data_encoded_improve.arff`: Improved version of encoded data
- `src.arff`: Source data file
- `src_removed.arff`: Processed data with removed attributes

## Implemented Classifiers
1. **Naive Bayes Classifier**
   - File: `naiveBayes_Classification.java`
   - Implementation of the Naive Bayes algorithm for classification

2. **OneR Classifier**
   - File: `oneR_Classification.java`
   - Implementation of the OneR (One Rule) algorithm

3. **J48 Decision Tree Classifier**
   - Directory: `J48_Classify_Result/`
   - Implementation of the J48 decision tree algorithm

## Requirements
- Java Development Kit (JDK)
- Weka Library (for machine learning algorithms)
- Required dependencies in the `lib/` directory

## Usage
1. Ensure all required dependencies are installed
2. Compile the Java files:
   ```bash
   javac -cp ".:lib/*" *.java
   ```
3. Run the classifiers:
   ```bash
   java -cp ".:lib/*" [ClassifierName]
   ```

## Results
Classification results are stored in the `results/` directory, organized by classifier type
