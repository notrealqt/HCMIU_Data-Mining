package Implementation.algorithms;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.File;

public class AssociationRules {
    public static void main(String[] args) {
        // Use all ARFF files in data/processed/reviews for Association Rules
        File reviewDir = new File("data/processed/reviews");
        File[] arffFiles = reviewDir.listFiles((dir, name) -> name.endsWith(".arff"));
        if (arffFiles == null) {
            System.err.println("No ARFF files found in data/processed/reviews");
            return;
        }
        for (File arffFile : arffFiles) {
            String datasetPath = arffFile.getPath();
            String fileName = arffFile.getName().replace(".arff", "");
            String outPath = "results/reports/association_rules_train_" + fileName + ".txt";
            try {
                // Load the ARFF file
                DataSource source = new DataSource(datasetPath);
                Instances data = source.getDataSet();

                // Initialize Apriori algorithm
                Apriori apriori = new Apriori();
                // You can set options here, e.g., min support/confidence
                // apriori.setOptions(new String[]{"-N", "10", "-C", "0.8", "-D", "0.05"});

                // Build associations and output the rules to a file
                try (PrintWriter out = new PrintWriter(new FileWriter(outPath))) {
                    out.println("=== Association Rules (Apriori) Training Results for " + fileName + " ===");
                    try {
                        apriori.buildAssociations(data);
                        out.println(apriori.toString());
                    } catch (Exception e) {
                        out.println("Association Rules could not be generated: " + e.getMessage());
                    }
                }
                System.out.println("Association Rules training results saved to " + outPath);

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
} 