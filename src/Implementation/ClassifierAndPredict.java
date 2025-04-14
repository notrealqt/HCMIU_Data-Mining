import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Random;

public class ClassifierAndPredict {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("src/Implementation/data.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); // Set class attribute

        // === Cross-validation evaluation ===
        Classifier tree = new J48();
        Evaluation evalCV = new Evaluation(data);
        evalCV.crossValidateModel(tree, data, 10, new Random(1));
        System.out.println("=== Cross-validation ===");
        System.out.println(evalCV.toSummaryString());

        // === Train/test split evaluation ===
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        Classifier cls = new J48();
        cls.buildClassifier(train);
        Evaluation evalTT = new Evaluation(train);
        evalTT.evaluateModel(cls, test);

        System.out.println("=== Train/Test Split (80/20) ===");
        System.out.println(evalTT.toSummaryString());

        // === Additional statistics ===
        System.out.println("Correctly Classified: " + evalTT.correct());
        System.out.println("Incorrectly Classified: " + evalTT.incorrect());
        System.out.println("Accuracy (%): " + evalTT.pctCorrect());
        System.out.println("Kappa: " + evalTT.kappa());
        System.out.println("MAE: " + evalTT.meanAbsoluteError());
        System.out.println("RMSE: " + evalTT.rootMeanSquaredError());
    }
}
