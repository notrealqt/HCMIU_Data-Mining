package Implementation.algorithms.J48_Classify;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.util.Random;

public class Tree_J48_Classifier {
    public static void main(String[] args) {
        try {
            // Load dataset
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(
                    "src/Implementation/algorithms/steam_game_data_encoded_improve.arff");
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

            // === Output results ===
            System.out.println("=== J48 Decision Tree Classifier Results ===");
            System.out.println("Classifier: \n" + tree.toString());
            System.out.println("\n=== Evaluation Summary ===");
            System.out.println("Accuracy: " + String.format("%.2f", eval.pctCorrect()) + "%");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
