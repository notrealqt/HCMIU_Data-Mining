import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class PredictionExample {
    public static void main(String[] args) throws Exception {
        // Load the trained model
        Classifier classifier = (Classifier) weka.core.SerializationHelper.read("trained_model.model");

        // Load new data to predict
        DataSource source = new DataSource("new_data.arff");
        Instances newData = source.getDataSet();
        newData.setClassIndex(newData.numAttributes() - 1); // Set the class attribute

        // Make predictions on new instances
        for (int i = 0; i < newData.numInstances(); i++) {
            Instance currentInstance = newData.instance(i);
            double predictedClass = classifier.classifyInstance(currentInstance);
            System.out.println("Predicted class for instance " + i + ": " + predictedClass);
        }
    }
}