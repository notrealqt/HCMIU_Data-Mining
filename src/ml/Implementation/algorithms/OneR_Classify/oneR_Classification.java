package Implementation.algorithms.OneR_Classify;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;
import java.io.FileWriter;
import java.io.PrintWriter;

public class oneR_Classification {
    public static void main(String[] args) {
        long totalStartTime = System.currentTimeMillis();
        // Only use the games ARFF files for OneR
        String[] arffFiles = {
            "data/processed/games/steam_game_data_encoded.arff", // before improvement
            "data/processed/games/steam_game_data_encoded_improve.arff" // after improvement
        };
        String[] tags = {"before", "after"};
        for (int i = 0; i < arffFiles.length; i++) {
            long startTime = System.currentTimeMillis();
            String datasetPath = arffFiles[i];
            String tag = tags[i];
            String outDir = "results/reports/" + tag + "/";
            try {
                DataSource source = new DataSource(datasetPath);
                Instances data = source.getDataSet();

                // remove the first two attributes
                String[] removeOptions = new String[] { "-R", "1,2" };
                Remove removeFilter = new Remove();
                removeFilter.setOptions(removeOptions);
                removeFilter.setInputFormat(data);
                Instances finalData = Filter.useFilter(data, removeFilter);

                // Set class attribute (all_reviews_encoded)
                finalData.setClassIndex(finalData.numAttributes() - 4);

                // Build OneR classifier
                OneR classifier = new OneR();
                classifier.setOptions(new String[] { "-B", "6" });
                long buildStartTime = System.currentTimeMillis();
                classifier.buildClassifier(finalData);
                long buildEndTime = System.currentTimeMillis();

                // Evaluate the model using 10-fold cross-validation
                Evaluation evaluation = new Evaluation(finalData);
                long evalStartTime = System.currentTimeMillis();
                evaluation.crossValidateModel(classifier, finalData, 10, new Random(1));
                long evalEndTime = System.currentTimeMillis();

                // Write results to file
                try (PrintWriter out = new PrintWriter(new FileWriter(outDir + "oneR_train.txt"))) {
                    out.println("=== OneR Classifier Training Results (" + tag + ") ===");
                    out.println("Selected Rule:\n" + classifier.toString());
                    out.println("\n=== Evaluation Summary ===");
                    out.println("Accuracy: " + String.format("%.2f%%", evaluation.pctCorrect()));
                    out.println(evaluation.toSummaryString());
                    out.println(evaluation.toClassDetailsString());
                    out.println(evaluation.toMatrixString());
                    
                    long endTime = System.currentTimeMillis();
                    out.printf("\nBuild time: %.2f seconds\n", (buildEndTime - buildStartTime) / 1000.0);
                    out.printf("Evaluation time: %.2f seconds\n", (evalEndTime - evalStartTime) / 1000.0);
                    out.printf("Total execution time: %.2f seconds\n", (endTime - startTime) / 1000.0);
                }
                System.out.println("OneR training results saved to " + outDir + "oneR_train.txt");

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        long totalEndTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (totalEndTime - totalStartTime) / 1000.0 + " seconds");
    }
} 