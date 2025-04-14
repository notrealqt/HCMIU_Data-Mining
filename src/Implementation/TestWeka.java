import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.packageManagement.Package;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;

public class TestWeka {
    public static void main(String[] args) throws Exception {
        // Load Weka packages (without GUI)
        WekaPackageManager.loadPackages(false);

        // // Print all available Weka packages and their versions
        // for (Package p : WekaPackageManager.getAllPackages()) {
        // System.out.println("- " + p.getName() + " / " +
        // p.getPackageMetaData().get("Version"));
        // }

        // for (Package p : WekaPackageManager.getInstalledPackages())
        // System.out.println("- " + p.getName() + "/" +
        // p.getPackageMetaData().get("Version"));
        DataSource source = new DataSource("./data_X.csv");
        // "./review/steam_game_reviews_part_1.csv");
        Instances data = source.getDataSet();

        String[] options = new String[1];
        options[0] = "-U";            // unpruned tree
        J48 tree = new J48();         // new instance of tree
        tree.setOptions(options);     // set the options
        tree.buildClassifier(data);   // build classifier
    }
}