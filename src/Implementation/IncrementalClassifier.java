import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.bayes.NaiveBayesUpdateable;

import java.io.File;

/**
 * This example trains NaiveBayes incrementally on data obtained
 * from the ArffLoader.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class IncrementalClassifier {

    /**
     * Expects an ARFF file as first argument (class attribute is assumed
     * to be the last attribute).
     *
     * @param args the commandline arguments
     * @throws Exception if something goes wrong
     */
    public static void main(String[] args) throws Exception {
        // load data
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(args[0]));
        Instances structure = loader.getStructure();
        structure.setClassIndex(structure.numAttributes() - 1);

        // train NaiveBayes
        NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
        nb.buildClassifier(structure);
        Instance current;
        while ((current = loader.getNextInstance(structure)) != null)
            nb.updateClassifier(current);

        // output generated model
        System.out.println(nb);
        // Load test data
        ArffLoader testLoader = new ArffLoader();
        testLoader.setFile(new File(args[1])); // Second argument: test ARFF file
        Instances testData = testLoader.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        // Predict and output results
        System.out.println("\n=== Predictions ===");
        for (int i = 0; i < testData.numInstances(); i++) {
            Instance testInst = testData.instance(i);
            double predIndex = nb.classifyInstance(testInst);
            String actual = testInst.stringValue(testData.classIndex());
            String predicted = testData.classAttribute().value((int) predIndex);
            System.out.printf("Instance %d: Actual: %s, Predicted: %s%n", i + 1, actual, predicted);
        }
    }
}