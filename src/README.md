Khoi:
Thien:
Trung:
Khoi:
Phat:
Tinh

# How to Run the Code

## Java (Machine Learning)

1. Make sure you have Java (JDK 8+) installed.
2. Download Weka and place `weka.jar` in the `lib/` directory (already present).
3. Compile all Java files from the `src/ml/Implementation/algorithms/` directory:

```sh
cd src/ml/Implementation/algorithms
javac -cp ../../../lib/weka.jar:. *.java J48_Classify/*.java
```

4. Run a Java class (example for OneR):

```sh
java -cp ../../../lib/weka.jar:. Implementation.algorithms.oneR_Classification
```

For J48:
```sh
cd J48_Classify
java -cp ../../../../lib/weka.jar:. Implementation.algorithms.J48_Classify.Tree_J48_Classifier
```

## Python (Data Crawling & Cleaning)

1. Install dependencies:
```sh
pip install requests pandas beautifulsoup4 urllib3
```
2. Run scripts from their respective directories so relative paths work:

```sh
cd src/data_crawling
python crawl_games.py
python crawl_reviews.py

cd ../data_cleaning
python clean_json.py
```

## Notes
- Make sure all required data files exist in the expected locations.
- Output files will be saved in the `results/reports/` directory.