import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

public class Classify {
    public static void main(String[] args) throws Exception {
        // Load training data
        DataSource source = new DataSource("train_data.arff");
        Instances trainingData = source.getDataSet();
        trainingData.setClassIndex(trainingData.numAttributes() - 1); // Set the class attribute

        // Train the classifier (J48)
        Classifier classifier = new J48();
        classifier.buildClassifier(trainingData);

        // Save the trained model
        SerializationHelper.write("trained_model.model", classifier);

        System.out.println("Model trained and saved successfully.");
    }
}
