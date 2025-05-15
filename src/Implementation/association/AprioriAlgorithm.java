package Implementation.association;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

public class AprioriAlgorithm {
    public static void main(String[] args) {
        try {
            // Load ARFF file
            DataSource source = new DataSource("src/Implementation/association/transactions.arff");
            Instances data = source.getDataSet();

            // Convert String attributes to Nominal
            StringToNominal filter = new StringToNominal();
            filter.setAttributeRange("first-last"); // Convert all attributes
            filter.setInputFormat(data);
            Instances filteredData = Filter.useFilter(data, filter);

            // Configure Apriori
            Apriori apriori = new Apriori();
            apriori.setClassIndex(filteredData.numAttributes() - 1);

            // Set parameters
            apriori.setNumRules(5);
            apriori.setLowerBoundMinSupport(0.01);
            apriori.setMinMetric(70.0);

            // Build associations
            apriori.buildAssociations(filteredData);

            // Output results
            System.out.println(apriori);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}