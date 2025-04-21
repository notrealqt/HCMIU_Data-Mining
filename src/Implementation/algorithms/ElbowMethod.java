package Implementation.algorithms;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ElbowMethod {
    public static void main(String[] args) throws Exception {
        // Load your dataset (ARFF or CSV)
        DataSource source = new DataSource("src/Implementation/algorithms/src.arff");
        Instances data = source.getDataSet();

        int maxK = 10;
        System.out.println("K\tSSE");
        for (int k = 1; k <= maxK; k++) {
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(k);
            kmeans.setSeed(10);
            kmeans.setPreserveInstancesOrder(true); // to compute SSE properly
            kmeans.buildClusterer(data);

            double sse = kmeans.getSquaredError();
            System.out.println(k + "\t" + sse);
        }
    }
}