// This file should be moved to src/ml/Implementation/algorithms/J48_Classify/Tree_J48_Classifier.java
// package Implementation.algorithms.J48_Classify;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.util.Random;
import java.io.FileWriter;
import java.io.PrintWriter;

public class Tree_J48_Classifier {
    public static void main(String[] args) {
        long totalStartTime = System.currentTimeMillis();
        // Only use the games ARFF files for J48
        String[] arffFiles = {
            "./data/processed/games/steam_game_data_encoded.arff", // before improvement
            "./data/processed/games/steam_game_data_encoded_improve.arff" // after improvement
        };
        String[] tags = {"before", "after"};
        for (int i = 0; i < arffFiles.length; i++) {
            long startTime = System.currentTimeMillis();
            String datasetPath = arffFiles[i];
            String tag = tags[i];
            String outDir = "./results/reports/" + tag + "/";
            try {
                ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetPath);
                Instances data = source.getDataSet();

                // === Keep only the selected attributes ===
                String[] keepOptions = new String[] { "-R", "3,4,5,6,7,8", "-V" };
                Remove keepFilter = new Remove();
                keepFilter.setOptions(keepOptions);
                keepFilter.setInputFormat(data);
                Instances filteredData = Filter.useFilter(data, keepFilter);

                // Set class attribute (by name or index)
                filteredData.setClassIndex(filteredData.attribute("all_reviews_encoded").index());

                // === Initialize J48 classifier ===
                J48 tree = new J48();
                tree.setOptions(new String[] { "-C", "0.5", "-M", "10" });

                // === Cross-validate ===
                Evaluation eval = new Evaluation(filteredData);
                eval.crossValidateModel(tree, filteredData, 10, new Random(1));

                // === Train on full dataset ===
                tree.buildClassifier(filteredData);

                // === Output results to file ===
                try (PrintWriter out = new PrintWriter(new FileWriter(outDir + "j48.txt"))) {
                    out.println("=== J48 Decision Tree Classifier Training Results (" + tag + ") ===");
                    out.println("Classifier: \n" + tree.toString());
                    out.println("\n=== Evaluation Summary ===");
                    out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
                    out.println(eval.toSummaryString());
                    out.println(eval.toClassDetailsString());
                    out.println(eval.toMatrixString());
                    
                    long endTime = System.currentTimeMillis();
                    out.printf("\nExecution time: %.2f seconds\n", (endTime - startTime) / 1000.0);
                }
                System.out.println("J48 training results saved to " + outDir + "j48.txt");

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        long totalEndTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (totalEndTime - totalStartTime) / 1000.0 + " seconds");
    }
} 