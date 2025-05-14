package test_and_eval.scripts;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;
import java.io.File;

public class ModelEvaluator {
    private static final String DATA_PATH = "../../src/Implementation/algorithms/steam_game_data_encoded.arff";
    private static final String RESULTS_PATH = "../results/";
    private static final String LEARNING_CURVE_CSV = "../results/learning_curve.csv";
    
    public static void main(String[] args) {
        try {
            // Load data
            System.out.println("Loading data from: " + DATA_PATH);
            DataSource source = new DataSource(DATA_PATH);
            Instances data = source.getDataSet();
            
            // Remove first two attributes
            String[] removeOptions = new String[]{"-R", "1,2"};
            Remove removeFilter = new Remove();
            removeFilter.setOptions(removeOptions);
            removeFilter.setInputFormat(data);
            Instances finalData = Filter.useFilter(data, removeFilter);
            
            // Set class attribute
            finalData.setClassIndex(finalData.numAttributes() - 4);
            
            // Learning curve data
            double[] trainFractions = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
            try (PrintWriter lcWriter = new PrintWriter(new FileWriter(LEARNING_CURVE_CSV))) {
                lcWriter.println("train_fraction,nb_train_acc,nb_test_acc,oner_train_acc,oner_test_acc,j48_train_acc,j48_test_acc");
                for (double frac : trainFractions) {
                    finalData.randomize(new Random(42));
                    int trainSize = (int) Math.round(finalData.numInstances() * frac);
                    int testSize = finalData.numInstances() - trainSize;
                    Instances trainData = new Instances(finalData, 0, trainSize);
                    Instances testData = new Instances(finalData, trainSize, testSize);

                    // Naive Bayes
                    NaiveBayes nb = new NaiveBayes();
                    nb.buildClassifier(trainData);
                    Evaluation nbTrainEval = new Evaluation(trainData);
                    nbTrainEval.evaluateModel(nb, trainData);
                    Evaluation nbTestEval = new Evaluation(trainData);
                    nbTestEval.evaluateModel(nb, testData);

                    // OneR
                    OneR oneR = new OneR();
                    oneR.setOptions(new String[]{"-B", "6"});
                    oneR.buildClassifier(trainData);
                    Evaluation oneRTrainEval = new Evaluation(trainData);
                    oneRTrainEval.evaluateModel(oneR, trainData);
                    Evaluation oneRTestEval = new Evaluation(trainData);
                    oneRTestEval.evaluateModel(oneR, testData);

                    // J48
                    J48 j48 = new J48();
                    j48.buildClassifier(trainData);
                    Evaluation j48TrainEval = new Evaluation(trainData);
                    j48TrainEval.evaluateModel(j48, trainData);
                    Evaluation j48TestEval = new Evaluation(trainData);
                    j48TestEval.evaluateModel(j48, testData);

                    lcWriter.printf("%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        frac,
                        nbTrainEval.pctCorrect(), nbTestEval.pctCorrect(),
                        oneRTrainEval.pctCorrect(), oneRTestEval.pctCorrect(),
                        j48TrainEval.pctCorrect(), j48TestEval.pctCorrect()
                    );
                }
            }
            System.out.println("Learning curve data saved to: " + LEARNING_CURVE_CSV);
            
            // Split data into training and test sets (80-20 split)
            finalData.randomize(new Random(1));
            int trainSize = (int) Math.round(finalData.numInstances() * 0.8);
            int testSize = finalData.numInstances() - trainSize;
            
            Instances trainData = new Instances(finalData, 0, trainSize);
            Instances testData = new Instances(finalData, trainSize, testSize);
            
            // Train and evaluate Naive Bayes
            System.out.println("\nEvaluating Naive Bayes...");
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(trainData);
            
            Evaluation nbEval = new Evaluation(trainData);
            nbEval.evaluateModel(nb, testData);
            
            // Train and evaluate OneR
            System.out.println("\nEvaluating OneR...");
            OneR oneR = new OneR();
            oneR.setOptions(new String[]{"-B", "6"});
            oneR.buildClassifier(trainData);
            
            Evaluation oneREval = new Evaluation(trainData);
            oneREval.evaluateModel(oneR, testData);
            
            // Train and evaluate J48
            System.out.println("\nEvaluating J48...");
            J48 j48 = new J48();
            j48.buildClassifier(trainData);
            
            Evaluation j48Eval = new Evaluation(trainData);
            j48Eval.evaluateModel(j48, testData);
            
            // Save results
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String resultFile = RESULTS_PATH + "evaluation_results_" + timestamp + ".txt";
            
            try (PrintWriter writer = new PrintWriter(new FileWriter(resultFile))) {
                writer.println("Model Evaluation Results");
                writer.println("======================");
                writer.println("Timestamp: " + new Date());
                writer.println("\nDataset Information:");
                writer.println("Total instances: " + finalData.numInstances());
                writer.println("Training instances: " + trainSize);
                writer.println("Test instances: " + testSize);
                writer.println("Number of attributes: " + finalData.numAttributes());
                writer.println("Number of classes: " + finalData.numClasses());
                
                writer.println("\nNaive Bayes Results:");
                writer.println("-------------------");
                writer.println("Accuracy: " + String.format("%.2f%%", nbEval.pctCorrect()));
                writer.println("Kappa statistic: " + String.format("%.4f", nbEval.kappa()));
                writer.println("Mean absolute error: " + String.format("%.4f", nbEval.meanAbsoluteError()));
                writer.println("Root mean squared error: " + String.format("%.4f", nbEval.rootMeanSquaredError()));
                writer.println("\nConfusion Matrix:");
                writer.println(nbEval.toMatrixString());
                
                writer.println("\nOneR Results:");
                writer.println("------------");
                writer.println("Accuracy: " + String.format("%.2f%%", oneREval.pctCorrect()));
                writer.println("Kappa statistic: " + String.format("%.4f", oneREval.kappa()));
                writer.println("Mean absolute error: " + String.format("%.4f", oneREval.meanAbsoluteError()));
                writer.println("Root mean squared error: " + String.format("%.4f", oneREval.rootMeanSquaredError()));
                writer.println("\nConfusion Matrix:");
                writer.println(oneREval.toMatrixString());
                
                writer.println("\nJ48 Results:");
                writer.println("------------");
                writer.println("Accuracy: " + String.format("%.2f%%", j48Eval.pctCorrect()));
                writer.println("Kappa statistic: " + String.format("%.4f", j48Eval.kappa()));
                writer.println("Mean absolute error: " + String.format("%.4f", j48Eval.meanAbsoluteError()));
                writer.println("Root mean squared error: " + String.format("%.4f", j48Eval.rootMeanSquaredError()));
                writer.println("\nConfusion Matrix:");
                writer.println(j48Eval.toMatrixString());
                
                writer.println("\nClass-wise Performance:");
                writer.println("---------------------");
                for (int i = 0; i < finalData.numClasses(); i++) {
                    writer.println("\nClass " + i + ":");
                    writer.println("Naive Bayes - Precision: " + String.format("%.4f", nbEval.precision(i)) +
                                 ", Recall: " + String.format("%.4f", nbEval.recall(i)) +
                                 ", F1: " + String.format("%.4f", nbEval.fMeasure(i)));
                    writer.println("OneR - Precision: " + String.format("%.4f", oneREval.precision(i)) +
                                 ", Recall: " + String.format("%.4f", oneREval.recall(i)) +
                                 ", F1: " + String.format("%.4f", oneREval.fMeasure(i)));
                    writer.println("J48 - Precision: " + String.format("%.4f", j48Eval.precision(i)) +
                                 ", Recall: " + String.format("%.4f", j48Eval.recall(i)) +
                                 ", F1: " + String.format("%.4f", j48Eval.fMeasure(i)));
                }
            }
            
            System.out.println("\nResults saved to: " + resultFile);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
} 