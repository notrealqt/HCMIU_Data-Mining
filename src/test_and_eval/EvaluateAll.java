package test_and_eval;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;
import java.io.FileWriter;
import java.io.PrintWriter;

public class EvaluateAll {
    public static void main(String[] args) {
        String[] arffFiles = {
            "data/processed/games/steam_game_data_encoded.arff", // before improvement
            "data/processed/games/steam_game_data_encoded_improve.arff" // after improvement
        };
        String[] tags = {"before", "after"};
        for (int i = 0; i < arffFiles.length; i++) {
            String datasetPath = arffFiles[i];
            String tag = tags[i];
            String outDir = "results/reports/" + tag + "/";
            System.out.println("\n=== Evaluating " + (tag.equals("before") ? "BEFORE" : "AFTER") + " improvement: " + datasetPath + " ===");
            try {
                DataSource source = new DataSource(datasetPath);
                Instances data = source.getDataSet();

                // Remove the first two attributes (if needed, as in other scripts)
                String[] removeOptions = new String[]{"-R", "1,2"};
                Remove removeFilter = new Remove();
                removeFilter.setOptions(removeOptions);
                removeFilter.setInputFormat(data);
                Instances filteredData = Filter.useFilter(data, removeFilter);

                // Set class attribute (as in previous scripts)
                filteredData.setClassIndex(filteredData.numAttributes() - 4);

                // --- Naive Bayes ---
                NaiveBayes nb = new NaiveBayes();
                nb.buildClassifier(filteredData);
                Evaluation evalNB = new Evaluation(filteredData);
                evalNB.crossValidateModel(nb, filteredData, 10, new Random(1));
                try (PrintWriter out = new PrintWriter(new FileWriter(outDir + "naive_bayes_eval.txt"))) {
                    out.println("=== Naive Bayes Classifier Results (" + tag + ") ===");
                    out.println("Accuracy: " + String.format("%.2f", evalNB.pctCorrect()) + "%");
                    out.println(evalNB.toSummaryString());
                    out.println(evalNB.toClassDetailsString());
                    out.println(evalNB.toMatrixString());
                }
                System.out.println("Naive Bayes results saved to " + outDir + "naive_bayes_eval.txt");

                // --- OneR ---
                OneR oneR = new OneR();
                oneR.setOptions(new String[]{"-B", "6"});
                oneR.buildClassifier(filteredData);
                Evaluation evalOneR = new Evaluation(filteredData);
                evalOneR.crossValidateModel(oneR, filteredData, 10, new Random(1));
                try (PrintWriter out = new PrintWriter(new FileWriter(outDir + "oneR_eval.txt"))) {
                    out.println("=== OneR Classifier Results (" + tag + ") ===");
                    out.println("Accuracy: " + String.format("%.2f", evalOneR.pctCorrect()) + "%");
                    out.println(evalOneR.toSummaryString());
                    out.println(evalOneR.toClassDetailsString());
                    out.println(evalOneR.toMatrixString());
                }
                System.out.println("OneR results saved to " + outDir + "oneR_eval.txt");

                // --- J48 ---
                J48 j48 = new J48();
                j48.setOptions(new String[]{"-C", "0.5", "-M", "10"});
                j48.buildClassifier(filteredData);
                Evaluation evalJ48 = new Evaluation(filteredData);
                evalJ48.crossValidateModel(j48, filteredData, 10, new Random(1));
                try (PrintWriter out = new PrintWriter(new FileWriter(outDir + "j48_eval.txt"))) {
                    out.println("=== J48 Decision Tree Classifier Results (" + tag + ") ===");
                    out.println("Accuracy: " + String.format("%.2f", evalJ48.pctCorrect()) + "%");
                    out.println(evalJ48.toSummaryString());
                    out.println(evalJ48.toClassDetailsString());
                    out.println(evalJ48.toMatrixString());
                }
                System.out.println("J48 results saved to " + outDir + "j48_eval.txt");

                // --- Association Rules (Apriori) ---
                Apriori apriori = new Apriori();
                try (PrintWriter out = new PrintWriter(new FileWriter(outDir + "association_rules_eval.txt"))) {
                    out.println("=== Association Rules (Apriori) (" + tag + ") ===");
                    try {
                        apriori.buildAssociations(data);
                        out.println(apriori.toString());
                    } catch (Exception e) {
                        out.println("Association Rules could not be generated: " + e.getMessage());
                    }
                }
                System.out.println("Association Rules results saved to " + outDir + "association_rules_eval.txt");

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
} 