package Implementation.algorithms;

import java.io.*;
import java.util.*;

public class ARFFBinaryToNominal {

    public static void main(String[] args) throws IOException {
        String inputFile = "src/Implementation/algorithms/cluster_2.arff";
        String outputFile = "src/Implementation/algorithms/output_2.arff";

        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));

        String line;
        boolean dataSection = false;

        // Define the attribute groups
        Map<String, Integer> attributeGroups = new LinkedHashMap<>();
        attributeGroups.put("price_encoded", 6);
        attributeGroups.put("release_year_encoded", 6);
        attributeGroups.put("all_reviews_encoded", 7);
        attributeGroups.put("all_percent_encoded", 6);
        attributeGroups.put("all_count_encoded", 7);
        attributeGroups.put("tags_encoded", 10);

        // Containers to store unique values and data
        Set<String> appIds = new TreeSet<>();
        Set<String> descriptions = new TreeSet<>();
        List<String> dataLines = new ArrayList<>();

        // First pass: collect values and store data lines
        while ((line = reader.readLine()) != null) {
            if (line.trim().equalsIgnoreCase("@data")) {
                dataSection = true;
                continue;
            }
            if (!dataSection || line.trim().isEmpty())
                continue;

            String[] values = splitCSV(line);
            if (values.length < 2)
                continue; // skip malformed lines

            appIds.add(values[0].trim());
            descriptions.add(values[1].trim());
            dataLines.add(line);
        }
        reader.close();

        // Write relation header
        writer.write("@relation steam_game_data_decoded");
        writer.newLine();
        writer.newLine();

        // Write collected app_ids
        writer.write("@attribute app_id {" + String.join(",", appIds) + "}");
        writer.newLine();

        // Write collected descriptions (quoted)
        List<String> quotedDescriptions = new ArrayList<>();
        // quote each description
        for (String desc : descriptions) {
            quotedDescriptions.add("'" + desc.replace("'", "") + "'");
        }
        writer.write("@attribute description {" + String.join(",", quotedDescriptions) + "}");
        writer.newLine();

        // Write new one-hot-decoded attributes
        for (Map.Entry<String, Integer> entry : attributeGroups.entrySet()) {
            writer.write(String.format("@attribute %s {%s}", entry.getKey(),
                    String.join(",", getRangeStrings(entry.getValue()))));
            writer.newLine();
        }

        writer.newLine();
        writer.write("@data");
        writer.newLine();

        // Process and write data section
        for (String dataLine : dataLines) {
            String[] values = splitCSV(dataLine);

            StringBuilder newLine = new StringBuilder();
            newLine.append(values[0]).append(",").append(values[1]);

            int idx = 2;
            for (Map.Entry<String, Integer> entry : attributeGroups.entrySet()) {
                int length = entry.getValue();
                int oneIndex = -1;

                for (int i = 0; i < length; i++) {
                    if (values.length > idx + i && values[idx + i].equals("1")) {
                        oneIndex = i;
                        break;
                    }
                }

                newLine.append(",").append(oneIndex);
                idx += length;
            }

            writer.write(newLine.toString());
            writer.newLine();
        }

        writer.close();
        System.out.println("ARFF transformation complete.");
    }

    // Helper to create range like 0,1,2,...
    private static List<String> getRangeStrings(int n) {
        List<String> range = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            range.add(String.valueOf(i));
        }
        return range;
    }

    // Properly split CSV lines, handling quoted values
    private static String[] splitCSV(String line) {
        List<String> result = new ArrayList<>();
        boolean inQuotes = false;
        StringBuilder current = new StringBuilder();

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (c == '\'') {
                inQuotes = !inQuotes;
                current.append(c); // keep quote
            } else if (c == ',' && !inQuotes) {
                result.add(current.toString().trim());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }

        result.add(current.toString().trim());
        return result.toArray(new String[0]);
    }
}
