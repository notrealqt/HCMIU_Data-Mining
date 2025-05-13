package Implementation.association;

import weka.associations.Apriori;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

import java.io.*;
import java.util.*;

public class AprioriAlgorithm {
    public static void main(String[] args) throws Exception {
        // Load original ARFF dataset
        Instances data = new DataSource("steam_game_review_encoded_part_1.arff").getDataSet();

        // Get indices for relevant attributes
        int appIdIdx = data.attribute("app_id").index();
        int votedUpIdx = data.attribute("voted_up").index();
        int steamIdIdx = data.attribute("steamid").index();

        // Step 1: Group app_ids by steamid where voted_up == 1
        Map<String, Set<String>> userTransactions = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            if (inst.value(votedUpIdx) == 1.0) {
                String user = inst.toString(steamIdIdx);
                String game = inst.toString(appIdIdx);
                userTransactions.computeIfAbsent(user, k -> new HashSet<>()).add(game);
            }
        }

        // Step 2: Create transactions ARFF
        File tempArff = new File("transactions.arff");
        try (PrintWriter writer = new PrintWriter(tempArff)) {
            writer.println("@RELATION transactions");
            writer.println("@ATTRIBUTE transaction STRING");
            writer.println("@DATA");
            for (Set<String> games : userTransactions.values()) {
                if (games.size() > 1) {
                    writer.println("'" + String.join(",", games) + "'");
                }
            }
        }

        // Step 3: Load transaction dataset
        Instances transactionData = new DataSource("transactions.arff").getDataSet();

        // Step 4: Convert string to nominal
        weka.filters.unsupervised.attribute.StringToWordVector filter = new weka.filters.unsupervised.attribute.StringToWordVector();
        filter.setInputFormat(transactionData);
        filter.setAttributeIndices("first"); // apply to transaction attribute
        Instances filteredData = Filter.useFilter(transactionData, filter);

        // Step 5: Run Apriori
        Apriori apriori = new Apriori();
        apriori.setLowerBoundMinSupport(2.0 / filteredData.numInstances()); // support >= 2
        apriori.setMinMetric(0.9); // confidence threshold (you can adjust)
        apriori.setNumRules(20); // number of rules to find

        apriori.buildAssociations(filteredData);

        // Output rules
        System.out.println(apriori);
    }
}
