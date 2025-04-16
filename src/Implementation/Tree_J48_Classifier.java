import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.J48;

import java.io.File;
import java.util.Random;
import weka.classifiers.evaluation.Evaluation;
import weka.core.SerializationHelper;

public class Tree_J48_Classifier {
    public void runJ48() throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("Test_DATA01.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(0);

        try {
            J48 tree = new J48();
            tree.setOptions(new String[] { "-C", "0.25", "-M", "2" });

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(tree, data, 10, new Random(1));
            tree.buildClassifier(data);

            System.out.println("=== J48 Model ===\n");
            System.out.println(tree);
            System.out.println(tree.graph());
            System.out.println("Correctly Classified: " + eval.correct());
            System.out.println("Incorrectly Classified: " + eval.incorrect());
            System.out.println("Accuracy (%): " + eval.pctCorrect());
            System.out.println("Kappa: " + eval.kappa());
            System.out.println("MAE: " + eval.meanAbsoluteError());
            System.out.println("RMSE: " + eval.rootMeanSquaredError());

            SerializationHelper.write("j48.bin", tree);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            Tree_J48_Classifier classifier = new Tree_J48_Classifier();
            classifier.runJ48();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
