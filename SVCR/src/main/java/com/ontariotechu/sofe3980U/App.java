package com.ontariotechu.sofe3980U;

import com.opencsv.*;
import java.io.FileReader;
import java.util.List;

/**
 * This program evaluates regression models based on error metrics and identifies the best model.
 */
public class App {
    public static void main(String[] args) {
        // CSV files containing model results
        String[] models = {"model_1.csv", "model_2.csv", "model_3.csv"};

        String optimalModel = "";
        double lowestMSE = Double.MAX_VALUE;

        // Iterate through each model to compute error metrics
        for (String model : models) {
            double mse = computeErrors(model);

            // Update best model based on MSE
            if (mse < lowestMSE) {
                lowestMSE = mse;
                optimalModel = model;
            }
        }

        // Display the model with the lowest error
        System.out.println("\nBest Model Based on MSE: " + optimalModel + " (MSE: " + String.format("%.6f", lowestMSE) + ")");
    }

    /**
     * Reads a CSV file to calculate Mean Squared Error (MSE), Mean Absolute Error (MAE),
     * and Mean Absolute Relative Error (MARE) using actual vs predicted values.
     *
     * @param filePath The dataset file path
     * @return The Mean Squared Error (MSE)
     */
    private static double computeErrors(String filePath) {
        List<String[]> dataset;

        try (FileReader reader = new FileReader(filePath);
             CSVReader csvReader = new CSVReaderBuilder(reader).withSkipLines(1).build()) {

            dataset = csvReader.readAll();
        } catch (Exception e) {
            System.out.println("Error processing file: " + filePath);
            return Double.MAX_VALUE;
        }

        int totalSamples = dataset.size();
        if (totalSamples == 0) {
            System.out.println("File is empty: " + filePath);
            return Double.MAX_VALUE;
        }

        double mse = 0, mae = 0, mare = 0;
        final double smallValue = 1e-10;
        int previewCount = Math.min(totalSamples, 10);

        System.out.println("\nProcessing: " + filePath);
        System.out.println("Actual\tPredicted");

        // Iterate through dataset rows
        for (int i = 0; i < totalSamples; i++) {
            String[] record = dataset.get(i);

            try {
                float actualValue = Float.parseFloat(record[0]);
                float predictedValue = Float.parseFloat(record[1]);

                double error = actualValue - predictedValue;
                mse += error * error;
                mae += Math.abs(error);
                mare += (Math.abs(error) / (Math.abs(actualValue) + smallValue));

                // Display a sample of values
                if (i < previewCount) {
                    System.out.println(actualValue + "\t" + predictedValue);
                }
            } catch (NumberFormatException ex) {
                System.out.println("Invalid data in file: " + filePath);
                return Double.MAX_VALUE;
            }
        }

        // Compute final error values
        mse /= totalSamples;
        mae /= totalSamples;
        mare /= totalSamples;

        // Output error metrics
        System.out.println("\nMSE: " + String.format("%.6f", mse));
        System.out.println("MAE: " + String.format("%.6f", mae));
        System.out.println("MARE: " + String.format("%.9f", mare));

        return mse;
    }
}
