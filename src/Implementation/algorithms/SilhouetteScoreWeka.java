package Implementation.algorithms;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.EuclideanDistance;
import weka.clusterers.SimpleKMeans;
import weka.core.converters.ConverterUtils.DataSource;

public class SilhouetteScoreWeka {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("src/Implementation/algorithms/src.arff");
        Instances data = source.getDataSet();

        int k = 3; // try different k values
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(k);
        kmeans.setPreserveInstancesOrder(true);
        kmeans.buildClusterer(data);

        int[] assignments = kmeans.getAssignments();
        EuclideanDistance distance = new EuclideanDistance(data);

        double[] silhouetteValues = new double[data.numInstances()];

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instanceI = data.instance(i);
            int clusterI = assignments[i];

            double a = 0.0;
            double b = Double.MAX_VALUE;
            int aCount = 0;

            // Calculate a(i) and b(i)
            for (int j = 0; j < data.numInstances(); j++) {
                if (i == j)
                    continue;
                Instance instanceJ = data.instance(j);
                int clusterJ = assignments[j];
                double dist = distance.distance(instanceI, instanceJ);

                if (clusterI == clusterJ) {
                    a += dist;
                    aCount++;
                } else {
                    // For b(i), track distances to other clusters
                    b = Math.min(b, dist); // Simplified (not exact average!)
                }
            }

            if (aCount > 0)
                a /= aCount;
            silhouetteValues[i] = (b - a) / Math.max(a, b);
        }

        // Compute average silhouette score
        double sum = 0.0;
        for (double s : silhouetteValues)
            sum += s;
        double avgSilhouette = sum / data.numInstances();

        System.out.printf("Silhouette Score for k=%d: %.4f\n", k, avgSilhouette);
    }
}
