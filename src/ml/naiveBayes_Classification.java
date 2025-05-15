package Implementation.algorithms;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;
import java.io.FileWriter;
import java.io.PrintWriter;

public class naiveBayes_Classification {
    public static void main(String[] args){
        // Only use the games ARFF files for Naive Bayes
        String[] arffFiles = {
            "data/processed/games/steam_game_data_encoded.arff", // before improvement
            "data/processed/games/steam_game_data_encoded_improve.arff" // after improvement
        };
        String[] tags = {"before", "after"};
        for (int i = 0; i < arffFiles.length; i++) {
            String datasetPath = arffFiles[i];
            String tag = tags[i];
            String outDir = "results/reports/" + tag + "/";
            try {
                ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetPath);
                Instances data = source.getDataSet();

                //remove the first two attributes
                String[] removeOptions = new String[]{"-R", "1,2"};
                Remove removeFilter = new Remove();
                removeFilter.setOptions(removeOptions);
                removeFilter.setInputFormat(data);
                Instances finalData = Filter.useFilter(data, removeFilter);
                // Set class attribute (all_reviews_encoded)
                finalData.setClassIndex(finalData.numAttributes() - 4);

                NaiveBayes classifier = new NaiveBayes();
                classifier.buildClassifier(finalData);

                Evaluation evaluation = new Evaluation(finalData);
                evaluation.crossValidateModel(classifier, finalData, 10, new Random(1));

                // Write results to file
                try (PrintWriter out = new PrintWriter(new FileWriter(outDir + "naive_bayes_train.txt"))) {
                    out.println("=== Naive Bayes Classifier Training Results (" + tag + ") ===");
                    out.println("Classifier: \n" + classifier.toString());
                    out.println("\n=== Evaluation Summary ===");
                    out.println("Accuracy: " + String.format("%.2f", evaluation.pctCorrect()) + "%");
                    out.println(evaluation.toSummaryString());
                    out.println(evaluation.toClassDetailsString());
                    out.println(evaluation.toMatrixString());
                }
                System.out.println("Naive Bayes training results saved to " + outDir + "naive_bayes_train.txt");

            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }
} 