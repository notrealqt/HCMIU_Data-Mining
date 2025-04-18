import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class PredictionExample {
    public static void main(String[] args) throws Exception {
        // Disable SecurityManager
        System.setSecurityManager(null); // Make sure to call this before anything else

        // Load the trained model
        Classifier classifier = (Classifier) weka.core.SerializationHelper.read("trained_model.model");

        // Load new data to predict
        DataSource source = new DataSource("Test_DATA02.csv");
        Instances newData = source.getDataSet();
        newData.setClassIndex(newData.numAttributes() - 1); // Set the class attribute (last column)

        // Make predictions on new instances
        for (int i = 0; i < newData.numInstances(); i++) {
            Instance currentInstance = newData.instance(i);
            double predictedClass = classifier.classifyInstance(currentInstance);

            // Convert predicted class to its string representation (for nominal classes)
            String predictedClassName = newData.classAttribute().value((int) predictedClass);
            System.out.println("Predicted class for instance " + i + ": " + predictedClassName);
        }
    }
}
