package Implementation.algorithms;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

import java.io.File;
import java.util.ArrayList;

public class Clustering {
    public static void main(String[] args) {
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/Implementation/algorithms/src.arff");
            Instances data = source.getDataSet();
            Instances encodedData = encodeNominalToBinary(data, "3-8");
            encodedData.setClassIndex(-1); // Unset class for unsupervised learning

            // Initialize SimpleKMeans clusterer
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(3); // Chosen using Elbow Method
            kmeans.setDistanceFunction(new EuclideanDistance());
            kmeans.setSeed(10);
            kmeans.setPreserveInstancesOrder(true);
            kmeans.buildClusterer(encodedData);

            // Print clustering summary
            // System.out.println("=== SimpleKMeans Clustering Results ===");
            // System.out.println(kmeans);

            // Evaluate clustering
            ClusterEvaluation clusterEvaluation = new ClusterEvaluation();
            clusterEvaluation.setClusterer(kmeans);
            clusterEvaluation.evaluateClusterer(encodedData);
            // System.out.println("\n=== Cluster Evaluation ===");
            // System.out.println(clusterEvaluation.clusterResultsToString());

            // === Extract and Save Each Cluster ===
            int numClusters = kmeans.getNumClusters();
            ArrayList<ArrayList<Instance>> clusterInstances = new ArrayList<>();

            for (int i = 0; i < numClusters; i++) {
                clusterInstances.add(new ArrayList<>());
            }

            // Assign instances to clusters
            for (int i = 0; i < encodedData.numInstances(); i++) {
                Instance instance = encodedData.instance(i);
                int clusterIndex = kmeans.clusterInstance(instance);
                clusterInstances.get(clusterIndex).add(instance);
            }

            // Convert each cluster to Instances and save to ARFF
            for (int i = 0; i < numClusters; i++) {
                Instances clusterData = new Instances(encodedData, clusterInstances.get(i).size());
                for (Instance inst : clusterInstances.get(i)) {
                    clusterData.add(inst);
                }

                // Save to ARFF
                ArffSaver saver = new ArffSaver();
                saver.setInstances(clusterData);
                saver.setFile(new File("src/Implementation/algorithms/cluster_" + i + ".arff"));
                saver.writeBatch();

                System.out.println("Saved Cluster " + i + " to cluster_" + i + ".arff");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Instances encodeNominalToBinary(Instances dataset, String attributeRange) throws Exception {
        NominalToBinary nominalToBinary = new NominalToBinary();
        nominalToBinary.setAttributeIndices(attributeRange);
        nominalToBinary.setInputFormat(dataset);
        return Filter.useFilter(dataset, nominalToBinary);
    }
}
