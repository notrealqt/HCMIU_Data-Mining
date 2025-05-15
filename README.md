# HCMIU Data Mining: Steam Game Review Analysis

## Overview
This project is a comprehensive data mining pipeline for analyzing Steam game data and user reviews. It covers data crawling, cleaning, preprocessing, feature engineering, and machine learning classification using both Python and Java (Weka). The goal is to predict review outcomes and extract insights from large-scale Steam data.

## Project Structure
```
HCMIU_Data-Mining/
│
├── README.md
├── requirements.txt / environment.yml   # Python dependencies (add as needed)
│
├── data/
│   ├── raw/
│   │   ├── games/                      # Raw game data (CSV, JSON)
│   │   └── reviews/                    # Raw review data (CSV, JSON)
│   ├── processed/
│   │   ├── games/                      # Cleaned/encoded game data (CSV, ARFF)
│   │   └── reviews/                    # Cleaned/encoded review data (CSV, ARFF)
│   └── external/                       # External datasets or reference files
│
├── notebooks/                          # Jupyter notebooks for EDA, cleaning, etc.
│
├── src/
│   ├── data_crawling/                  # Python scripts for crawling Steam data
│   ├── data_cleaning/                  # Data cleaning scripts
│   ├── preprocessing/                  # Feature engineering and preprocessing
│   ├── features/                       # (Optional) Feature engineering scripts
│   └── ml/                             # Java (Weka) ML training scripts
│
├── models/                             # Trained model files
│
├── results/
│   ├── figures/                        # Visualizations (e.g., decision trees)
│   ├── reports/                        # Experiment results, logs, summaries
│   └── logs/                           # (Optional) Log files
│
├── lib/                                # External libraries (e.g., Weka JARs)
│
└── scripts/                            # Helper scripts (e.g., for evaluation)
```

## Key Components
- **Data Crawling:** Python scripts to scrape game and review data from Steam.
- **Data Cleaning & Preprocessing:** Jupyter notebooks and scripts for cleaning, merging, and encoding data.
- **Feature Engineering:** (Optional) Scripts for advanced feature extraction.
- **Machine Learning:** Java (Weka) scripts for training and evaluating classifiers (Naive Bayes, OneR, J48 Decision Tree).
- **Results:** Evaluation reports, figures, and trained models.

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/HCMIU_Data-Mining.git
   cd HCMIU_Data-Mining
   ```
2. **Install Python dependencies:**
   - Create a virtual environment (recommended)
   - Install packages from `requirements.txt` or `environment.yml` (to be created)

3. **Install Java and Weka:**
   - Ensure Java is installed (`java -version`)
   - Weka JARs are included in `lib/`

4. **Run Data Crawling:**
   - Use scripts in `src/data_crawling/` to fetch raw data from Steam (requires internet access)

5. **Data Cleaning & Preprocessing:**
   - Use notebooks in `notebooks/` to clean and preprocess data
   - Processed data will be saved in `data/processed/`

6. **Machine Learning:**
   - Compile and run Java files in `src/ml/` using Weka and the processed ARFF files
   - Example:
     ```bash
     javac -cp lib/weka.jar src/ml/train_j48.java
     java -cp lib/weka.jar:src/ml train_j48
     ```

7. **View Results:**
   - Check `results/reports/` and `results/figures/` for evaluation outputs and visualizations

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License
[MIT License](LICENSE) (or your preferred license)

## Authors
- Khoi
- Thien
- Trung
- Phat
- Tinh

## Acknowledgements
- Steam for providing public data
- Weka for machine learning tools
- HCMIU Data Mining course