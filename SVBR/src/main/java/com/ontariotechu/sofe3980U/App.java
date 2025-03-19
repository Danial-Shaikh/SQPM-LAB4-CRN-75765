package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.*;
import com.opencsv.*;

/**
 * Evaluate Single Variable Binary Regression
 */
public class App
{
	public static void main(String[] args) {
		// Define model file names
		String[] modelFiles = {"model_1.csv", "model_2.csv", "model_3.csv"};
		String bestModel = "";
		double bestAUC = 0.0;

		for (String filePath : modelFiles) {
			System.out.println("Evaluating " + filePath);
			double auc = evaluateModel(filePath);
			if (auc > bestAUC) {
				bestAUC = auc;
				bestModel = filePath;
			}
			System.out.println("---------------------------------------------");
		}

		// Display the best model based on AUC-ROC
		System.out.println("Best performing model: " + bestModel + " with AUC-ROC = " + bestAUC);
	}

	public static double evaluateModel(String filePath) {
		List<String[]> allData;
		try {
			FileReader filereader = new FileReader(filePath);
			CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
			allData = csvReader.readAll();
		} catch (Exception e) {
			System.out.println("Error reading the CSV file: " + filePath);
			return 0.0;
		}

		int TP = 0, FP = 0, TN = 0, FN = 0;
		double BCE = 0.0;
		int n = allData.size();
		double threshold = 0.5;
		List<Double> y_true_list = new ArrayList<>();
		List<Double> y_pred_list = new ArrayList<>();

		// Process each row in the CSV file
		for (String[] row : allData) {
			double y_true = Double.parseDouble(row[0]);
			double y_pred = Double.parseDouble(row[1]);

			// Convert predicted value to binary
			int y_pred_binary = (y_pred >= threshold) ? 1 : 0;

			// Compute confusion matrix values
			if (y_true == 1 && y_pred_binary == 1) TP++;
			else if (y_true == 0 && y_pred_binary == 1) FP++;
			else if (y_true == 0 && y_pred_binary == 0) TN++;
			else if (y_true == 1 && y_pred_binary == 0) FN++;

			// Compute Binary Cross-Entropy
			BCE += y_true * Math.log(y_pred + 1e-9) + (1 - y_true) * Math.log(1 - y_pred + 1e-9);

			// Store values for AUC-ROC computation
			y_true_list.add(y_true);
			y_pred_list.add(y_pred);
		}

		BCE = -BCE / n; // Final BCE calculation
		double accuracy = (double) (TP + TN) / (TP + TN + FP + FN);
		double precision = (TP + FP) > 0 ? (double) TP / (TP + FP) : 0;
		double recall = (TP + FN) > 0 ? (double) TP / (TP + FN) : 0;
		double f1_score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
		double auc_roc = calculateAUC(y_true_list, y_pred_list);

		// Print results
		System.out.println("BCE: " + BCE);
		System.out.println("Confusion Matrix: TP=" + TP + " FP=" + FP + " TN=" + TN + " FN=" + FN);
		System.out.println("Accuracy: " + accuracy);
		System.out.println("Precision: " + precision);
		System.out.println("Recall: " + recall);
		System.out.println("F1 Score: " + f1_score);
		System.out.println("AUC-ROC: " + auc_roc);

		return auc_roc;
	}

	// Function to calculate AUC-ROC
	public static double calculateAUC(List<Double> y_true, List<Double> y_pred) {
		int n = y_true.size();
		int n_positive = (int) y_true.stream().filter(y -> y == 1).count();
		int n_negative = n - n_positive;
		List<Double> thresholds = new ArrayList<>();
		for (int i = 0; i <= 100; i++) {
			thresholds.add(i / 100.0);
		}

		List<Double> tpr = new ArrayList<>();
		List<Double> fpr = new ArrayList<>();

		for (double th : thresholds) {
			int TP = 0, FP = 0, FN = 0, TN = 0;
			for (int i = 0; i < n; i++) {
				int pred_binary = (y_pred.get(i) >= th) ? 1 : 0;
				if (y_true.get(i) == 1 && pred_binary == 1) TP++;
				else if (y_true.get(i) == 0 && pred_binary == 1) FP++;
				else if (y_true.get(i) == 0 && pred_binary == 0) TN++;
				else if (y_true.get(i) == 1 && pred_binary == 0) FN++;
			}
			tpr.add(n_positive > 0 ? (double) TP / n_positive : 0);
			fpr.add(n_negative > 0 ? (double) FP / n_negative : 0);
		}

		// Compute AUC using the trapezoidal rule
		double auc = 0.0;
		for (int i = 1; i < tpr.size(); i++) {
			auc += (tpr.get(i - 1) + tpr.get(i)) * Math.abs(fpr.get(i - 1) - fpr.get(i)) / 2;
		}
		return auc;
	}
}