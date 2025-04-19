package Implementation.algorithms;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class oneR_Classification {
    public static void main(String[] args) {
        try {
            // Step 1: Load the arff file
            DataSource source = new DataSource("src/Implementation/algorithms/src.arff");
            Instances data = source.getDataSet();

            //remove the first two attributes
            String[] removeOptions = new String[]{"-R", "1,2"};
            Remove removeFilter = new Remove();
            removeFilter.setOptions(removeOptions);
            removeFilter.setInputFormat(data);
            Instances finalData = Filter.useFilter(data, removeFilter);



            // Step 4: Set class attribute (all_reviews_encoded)
            finalData.setClassIndex(finalData.numAttributes() - 4); // Last column

            // Step 5: Build OneR classifier
            OneR classifier = new OneR();
            classifier.setOptions(new String[]{"-B", "6"}); // Minimum bucket size for discretization
            classifier.buildClassifier(finalData);

            // Step 6: Evaluate the model using 10-fold cross-validation
            Evaluation evaluation = new Evaluation(finalData);
            evaluation.crossValidateModel(classifier, finalData, 10, new Random(1));

            // Step 7: Print results
            System.out.println("=== OneR Classifier Results ===");
            System.out.println("Selected Rule:\n" + classifier.toString());
            System.out.println("\n=== Evaluation Summary ===");
            System.out.println("Accuracy: " + String.format("%.2f%%", evaluation.pctCorrect()));
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}