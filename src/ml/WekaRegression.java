import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

public class WekaRegression {
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        String[] arffFiles = {
            "data/processed/games/steam_game_data_encoded.arff", // before improvement
            "data/processed/games/steam_game_data_encoded_improve.arff" // after improvement
        };
        String[] tags = {"before", "after"};
        String[] modelNames = {"linear_regression", "m5p", "smoreg"};
        String testFile = "data/processed/games/model_test.arff";

        for (int i = 0; i < arffFiles.length; i++) {
            String datasetPath = arffFiles[i];
            String tag = tags[i];
            String outDir = "results/reports/" + tag + "/";
            try {
                System.out.println("Loading dataset: " + datasetPath);
                ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetPath);
                Instances data = source.getDataSet();
                System.out.println("Dataset loaded successfully. Number of instances: " + data.numInstances());

                // Set class attribute first (price_encoded)
                data.setClassIndex(2); // price_encoded is the 3rd attribute (0-based index)
                System.out.println("Class attribute set to: " + data.attribute(2).name());

                // Print attribute info and unique value counts
                printAttributeInfo(data);

                // Remove constant columns
                Instances filteredData = removeConstantAttributes(data);
                System.out.println("Attributes after removing constants: " + filteredData.numAttributes());

                // Check class attribute
                Attribute classAttr = filteredData.classAttribute();
                if (!classAttr.isNumeric()) {
                    System.err.println("Class attribute is not numeric! Skipping regression.");
                    writeErrorToAllModels(outDir, modelNames, tag, "Class attribute is not numeric!");
                    continue;
                }
                Set<Double> classVals = new HashSet<>();
                for (int j = 0; j < filteredData.numInstances(); j++) {
                    classVals.add(filteredData.instance(j).classValue());
                }
                if (classVals.size() < 2) {
                    System.err.println("Class attribute has less than 2 unique values! Skipping regression.");
                    writeErrorToAllModels(outDir, modelNames, tag, "Class attribute has less than 2 unique values!");
                    continue;
                }

                // Prepare test set
                System.out.println("Loading test dataset: " + testFile);
                ConverterUtils.DataSource testSource = new ConverterUtils.DataSource(testFile);
                Instances testData = testSource.getDataSet();
                testData.setClassIndex(2); // Set class index before removing attributes
                Instances filteredTestData = removeConstantAttributes(testData);
                System.out.println("Test attributes after removing constants: " + filteredTestData.numAttributes());

                // Run all three regression models
                System.out.println("\nRunning regression models for " + tag + " dataset:");
                runAndSaveRegression(new LinearRegression(), filteredData, filteredTestData, outDir + modelNames[0] + ".txt", "Linear Regression", tag);
                runAndSaveRegression(new M5P(), filteredData, filteredTestData, outDir + modelNames[1] + ".txt", "M5P", tag);
                runAndSaveRegression(new SMOreg(), filteredData, filteredTestData, outDir + modelNames[2] + ".txt", "SMOreg", tag);

            } catch (Exception e) {
                System.err.println("Error processing dataset " + datasetPath + ":");
                e.printStackTrace();
                writeErrorToAllModels(outDir, modelNames, tag, e.getMessage());
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime) / 1000.0 + " seconds");
    }

    private static void printAttributeInfo(Instances data) {
        System.out.println("Attributes:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            Set<Double> uniqueVals = new HashSet<>();
            for (int j = 0; j < data.numInstances(); j++) {
                uniqueVals.add(data.instance(j).value(attr));
                if (uniqueVals.size() > 10) break;
            }
            System.out.println("  - " + attr.name() + " (" + (attr.isNumeric() ? "numeric" : "nominal") + ", unique sample: " + uniqueVals.size() + ")");
        }
    }

    private static Instances removeConstantAttributes(Instances data) throws Exception {
        List<Integer> toRemove = new ArrayList<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i == data.classIndex()) continue;
            Attribute attr = data.attribute(i);
            Set<Double> uniqueVals = new HashSet<>();
            for (int j = 0; j < data.numInstances(); j++) {
                uniqueVals.add(data.instance(j).value(attr));
                if (uniqueVals.size() > 1) break;
            }
            if (uniqueVals.size() <= 1) {
                toRemove.add(i + 1); // Weka Remove filter is 1-based
                System.out.println("Removing constant attribute: " + attr.name());
            }
        }
        if (toRemove.isEmpty()) return data;
        StringBuilder idxs = new StringBuilder();
        for (int idx : toRemove) {
            if (idxs.length() > 0) idxs.append(",");
            idxs.append(idx);
        }
        Remove remove = new Remove();
        remove.setOptions(new String[]{"-R", idxs.toString()});
        remove.setInputFormat(data);
        return Filter.useFilter(data, remove);
    }

    private static void writeErrorToAllModels(String outDir, String[] modelNames, String tag, String message) {
        for (String model : modelNames) {
            try (PrintWriter out = new PrintWriter(new FileWriter(outDir + model + ".txt"))) {
                out.println("=== " + model + " Regression Results (" + tag + ") ===");
                out.println("ERROR: " + message);
            } catch (Exception ignored) {}
        }
    }

    private static void runAndSaveRegression(weka.classifiers.Classifier model, Instances data, Instances testData, String outPath, String modelName, String tag) {
        long startTime = System.currentTimeMillis();
        try (PrintWriter out = new PrintWriter(new FileWriter(outPath))) {
            System.out.println("Building " + modelName + " model...");
            out.println("=== " + modelName + " Regression Results (" + tag + ") ===");
            model.buildClassifier(data);
            out.println("\nModel Summary:\n" + model.toString());

            System.out.println("Performing cross-validation...");
            // Evaluate with 10-fold cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));

            out.println("\n=== Evaluation Summary (Train/CV) ===");
            out.printf("RMSE: %.4f\n", eval.rootMeanSquaredError());
            out.printf("MAE: %.4f\n", eval.meanAbsoluteError());
            out.printf("R2 (Correlation Coefficient^2): %.4f\n", Math.pow(eval.correlationCoefficient(), 2));
            out.println(eval.toSummaryString());
            out.println(eval.toClassDetailsString());
            out.println(eval.toMatrixString());

            System.out.println("Evaluating on test set...");
            // Test set evaluation
            out.println("\n=== Test Set Evaluation ===");
            double sumSqErr = 0, sumAbsErr = 0, sumY = 0, sumY2 = 0, sumPred = 0, sumPred2 = 0, sumYPred = 0;
            int n = testData.numInstances();
            for (int i = 0; i < n; i++) {
                double actual = testData.instance(i).classValue();
                double pred = model.classifyInstance(testData.instance(i));
                sumSqErr += Math.pow(actual - pred, 2);
                sumAbsErr += Math.abs(actual - pred);
                sumY += actual;
                sumY2 += actual * actual;
                sumPred += pred;
                sumPred2 += pred * pred;
                sumYPred += actual * pred;
            }
            double rmse = Math.sqrt(sumSqErr / n);
            double mae = sumAbsErr / n;
            double r2 = pearson(sumY, sumY2, sumPred, sumPred2, sumYPred, n);
            out.printf("Test RMSE: %.4f\n", rmse);
            out.printf("Test MAE: %.4f\n", mae);
            out.printf("Test R2: %.4f\n", r2);

            long endTime = System.currentTimeMillis();
            out.printf("\nExecution time: %.2f seconds\n", (endTime - startTime) / 1000.0);
            System.out.println(modelName + " results saved to " + outPath);
        } catch (Exception e) {
            System.err.println("Error running " + modelName + ":");
            e.printStackTrace();
            try (PrintWriter out = new PrintWriter(new FileWriter(outPath))) {
                out.println("=== " + modelName + " Regression Results (" + tag + ") ===");
                out.println("ERROR: " + e.getMessage());
            } catch (Exception ignored) {}
        }
    }

    // Pearson correlation helper
    private static double pearson(double sumY, double sumY2, double sumPred, double sumPred2, double sumYPred, int n) {
        double num = n * sumYPred - sumY * sumPred;
        double den = Math.sqrt((n * sumY2 - sumY * sumY) * (n * sumPred2 - sumPred * sumPred));
        return den == 0 ? 0 : Math.pow(num / den, 2); // R^2
    }
} 