#!/bin/bash

# Create package directory
mkdir -p test_and_eval/scripts

# Compile the test script
javac -cp "../../lib/weka.jar" -d . ModelEvaluator.java

# Run the evaluation
java -cp "../../lib/weka.jar:." test_and_eval.scripts.ModelEvaluator 