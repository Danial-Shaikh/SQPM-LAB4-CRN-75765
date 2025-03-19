package com.ontariotechu.sofe3980U;

import com.opencsv.*;
import java.io.FileReader;
import java.util.List;

/**
 * This program evaluates multiple regression models based on their Mean Squared Error (MSE)
 * and recommends the best-performing model.
 */
public class App {
	public static void main(String[] args) {
		// List of CSV files containing model predictions
		String[] modelFiles = {"model_1.csv", "model_2.csv", "model_3.csv"};

		String bestModel = "";
		double minError = Double.MAX_VALUE;

		// Iterate through each model file to evaluate errors
		for (String modelFile : modelFiles) {
			double mse = evaluateModel(modelFile);

			// Update best model if a lower MSE is found
			if (mse < minError) {
				minError = mse;
				bestModel = modelFile;
			}
		}

		// Display the recommended model based on the lowest error
		System.out.println("\nBest Model: " + bestModel + " (MSE: " + String.format("%.5f", minError) + ")");
	}

	/**
	 * Reads a CSV file and calculates Mean Squared Error (MSE), Mean Absolute Error (MAE),
	 * and Mean Absolute Relative Error (MARE) based on true vs predicted values.
	 *
	 * @param filePath The CSV file path containing true and predicted values
	 * @return The calculated Mean Squared Error (MSE)
	 */
	private static double evaluateModel(String filePath) {
		List<String[]> data;

		try (FileReader fileReader = new FileReader(filePath);
			 CSVReader csvReader = new CSVReaderBuilder(fileReader).withSkipLines(1).build()) {

			data = csvReader.readAll();
		} catch (Exception e) {
			System.out.println("Error reading file: " + filePath);
			return Double.MAX_VALUE; // Return high error for faulty files
		}

		int sampleCount = data.size();
		if (sampleCount == 0) {
			System.out.println("Empty file: " + filePath);
			return Double.MAX_VALUE;
		}

		double mse = 0, mae = 0, mare = 0;
		double epsilon = 1e-10; // Small value to prevent division by zero
		int displayLimit = Math.min(sampleCount, 10);

		System.out.println("\nEvaluating " + filePath + "...");
		System.out.println("True Value\tPredicted Value");

		// Process each row in the dataset
		for (int i = 0; i < sampleCount; i++) {
			String[] row = data.get(i);

			try {
				float actual = Float.parseFloat(row[0]);
				float predicted = Float.parseFloat(row[1]);

				double error = actual - predicted;
				mse += error * error;
				mae += Math.abs(error);
				mare += (Math.abs(error) / (Math.abs(actual) + epsilon)) * 100;

				// Display first few samples
				if (i < displayLimit) {
					System.out.println(actual + "\t" + predicted);
				}
			} catch (NumberFormatException ex) {
				System.out.println("Invalid data format in file: " + filePath);
				return Double.MAX_VALUE;
			}
		}

		// Compute final error metrics
		mse /= sampleCount;
		mae /= sampleCount;
		mare /= sampleCount;

		// Print error metrics
		System.out.println("\nMSE: " + String.format("%.5f", mse));
		System.out.println("MAE: " + String.format("%.5f", mae));
		System.out.println("MARE: " + String.format("%.5f", mare) + " %\n");

		return mse;
	}
}