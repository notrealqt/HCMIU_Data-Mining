package Implementation.algorithms;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class naiveBayes_Classification {
    public static void main(String[] args){
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/Implementation/algorithms/src.arff");
            Instances data = source.getDataSet();

            //remove the first two attributes
            String[] removeOptions = new String[]{"-R", "1,2"};
            Remove removeFilter = new Remove();
            removeFilter.setOptions(removeOptions);
            removeFilter.setInputFormat(data);
            Instances finalData = Filter.useFilter(data, removeFilter);
            // Step 4: Set class attribute (all_reviews_encoded)
            finalData.setClassIndex(finalData.numAttributes() - 4);

            NaiveBayes classifier = new NaiveBayes();
            classifier.buildClassifier(finalData);

            Evaluation evaluation = new Evaluation(finalData);
            evaluation.crossValidateModel(classifier, finalData, 10, new Random(1));

            System.out.println("=== Naive Bayes Classifier Results ===");
            System.out.println("Classifier: \n" + classifier.toString());
            System.out.println("\n=== Evaluation Summary ===");
            System.out.println("Accuracy: " + String.format("%.2f", evaluation.pctCorrect()) + "%");
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());

        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
