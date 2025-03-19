package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.List;
import com.opencsv.*;
import java.util.Arrays;

/**
 * Evaluates Multiclass Classification Model
 * Computes Cross-Entropy loss and Confusion Matrix from CSV input
 */
public class App {
    public static void main(String[] args) {
        String filePath = "model.csv";
        List<String[]> data;
        
        try (FileReader fileReader = new FileReader(filePath);
             CSVReader csvReader = new CSVReaderBuilder(fileReader).withSkipLines(1).build()) {
            data = csvReader.readAll();
        } catch (Exception e) {
            System.err.println("Error: Unable to read CSV file.");
            return;
        }

        final int CLASS_COUNT = 5; // Assuming classification into 5 classes
        int totalSamples = data.size();
        double crossEntropyLoss = 0.0;
        int[][] confusionMatrix = new int[CLASS_COUNT][CLASS_COUNT];

        for (String[] row : data) {
            int actualClass = Integer.parseInt(row[0]) - 1; // Convert to zero-based index
            float[] predictedProbs = new float[CLASS_COUNT];

            for (int i = 0; i < CLASS_COUNT; i++) {
                predictedProbs[i] = Float.parseFloat(row[i + 1]);
            }

            int predictedClass = getPredictedClass(predictedProbs);
            confusionMatrix[actualClass][predictedClass]++;
            crossEntropyLoss += Math.log(predictedProbs[actualClass] + 1e-9); // Avoid log(0)
        }

        crossEntropyLoss = -crossEntropyLoss / totalSamples;

        displayResults(crossEntropyLoss, confusionMatrix);
    }

    private static int getPredictedClass(float[] probabilities) {
        int maxIndex = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static void displayResults(double ceLoss, int[][] matrix) {
        System.out.printf("CE =%.7f\n", ceLoss);
        System.out.println("Confusion matrix");
        System.out.println("\t\ty=1\ty=2\ty=3\ty=4\ty=5");
        for (int i = 0; i < matrix.length; i++) {
            System.out.printf("y^=%d", i + 1);
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.printf("\t%d", matrix[i][j]);
            }
            System.out.println();
        }
    }
}
