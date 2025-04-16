import weka.core.Instances;
import weka.classifiers.Classifier;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

public class UnlabeledDataClassify {
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("Usage: java UnlabeledDataClassify <modelPath> <unlabeled.arff> <output.arff>");
            return;
        }

        String modelPath = args[0];
        String unlabeledPath = args[1];
        String outputPath = args[2];

        // Load unlabeled data
        Instances unlabeled = new Instances(new BufferedReader(new FileReader(unlabeledPath)));
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

        // Load trained classifier (previously saved model)
        Classifier classifier = (Classifier) weka.core.SerializationHelper.read(modelPath);

        // Create copy of data for labeled output
        Instances labeled = new Instances(unlabeled);

        // Classify each instance and label
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = classifier.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }

        // Save labeled data to new ARFF
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));
        writer.write(labeled.toString());
        writer.newLine();
        writer.flush();
        writer.close();

        System.out.println("Classification complete. Labeled data written to: " + outputPath);
    }
}
