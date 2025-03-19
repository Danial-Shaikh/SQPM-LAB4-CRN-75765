package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.List;
import com.opencsv.*;
import java.util.Arrays;

/**
 * Multiclass Classification Evaluation
 * Computes Cross-Entropy loss and Confusion Matrix from CSV data
 */
public class App {
	public static void main(String[] args) {
		String filePath = "model.csv";
		FileReader filereader;
		List<String[]> allData;

		try {
			// Read CSV file while skipping the header
			filereader = new FileReader(filePath);
			CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
			allData = csvReader.readAll();
		} catch (Exception e) {
			System.out.println("Error reading the CSV file");
			return;
		}

		int numClasses = 5; // Assuming 5 classes (1 to 5)
		int n = allData.size(); // Number of samples
		double crossEntropy = 0.0;
		int[][] confusionMatrix = new int[numClasses][numClasses]; // Confusion matrix

		// Process each row in the CSV
		for (String[] row : allData) {
			int y_true = Integer.parseInt(row[0]) - 1; // Convert class index to 0-based
			float[] y_predicted = new float[numClasses];

			// Read predicted probabilities
			for (int i = 0; i < numClasses; i++) {
				y_predicted[i] = Float.parseFloat(row[i + 1]);
			}

			// Find predicted class (index of max probability)
			int y_pred = 0;
			for (int i = 1; i < numClasses; i++) {
				if (y_predicted[i] > y_predicted[y_pred]) {
					y_pred = i;
				}
			}

			// Update confusion matrix
			confusionMatrix[y_true][y_pred]++;

			// Compute cross-entropy loss
			crossEntropy += Math.log(y_predicted[y_true]);
		}

		// Compute final cross-entropy loss
		crossEntropy = -crossEntropy / n;

		// Display results
		System.out.println("Cross-Entropy Loss: " + crossEntropy);
		System.out.println("Confusion Matrix:");
		for (int i = 0; i < numClasses; i++) {
			System.out.println(Arrays.toString(confusionMatrix[i]));
		}
	}
}