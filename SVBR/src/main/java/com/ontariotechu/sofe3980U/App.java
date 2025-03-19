package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.*;
import com.opencsv.*;

public class App {
	public static void main(String[] args) {
		String[] modelFiles = {"model_1.csv", "model_2.csv", "model_3.csv"};
		Map<String, Double> metrics = new HashMap<>();

		for (String file : modelFiles) {
			System.out.println("for " + file);
			double auc = processModel(file);
			metrics.put(file, auc);
			System.out.println();
		}

		determineBestModel(metrics);
	}

	private static double processModel(String fileName) {
		List<String[]> records;
		try (FileReader reader = new FileReader(fileName);
			 CSVReader csvReader = new CSVReaderBuilder(reader).withSkipLines(1).build()) {
			records = csvReader.readAll();
		} catch (Exception e) {
			System.err.println("Error reading file: " + fileName);
			return 0.0;
		}

		int TP = 0, FP = 0, TN = 0, FN = 0;
		double BCE = 0.0;
		int total = records.size();
		double threshold = 0.5;
		List<Double> actuals = new ArrayList<>(), predictions = new ArrayList<>();

		for (String[] record : records) {
			double actual = Double.parseDouble(record[0]);
			double predicted = Double.parseDouble(record[1]);
			int predictedLabel = (predicted >= threshold) ? 1 : 0;

			if (actual == 1) {
				if (predictedLabel == 1) TP++; else FN++;
			} else {
				if (predictedLabel == 1) FP++; else TN++;
			}

			BCE += actual * Math.log(predicted + 1e-9) + (1 - actual) * Math.log(1 - predicted + 1e-9);
			actuals.add(actual);
			predictions.add(predicted);
		}

		BCE = -BCE / total;
		double accuracy = (double) (TP + TN) / total;
		double precision = (TP + FP) > 0 ? (double) TP / (TP + FP) : 0;
		double recall = (TP + FN) > 0 ? (double) TP / (TP + FN) : 0;
		double f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
		double aucRoc = computeAUC(actuals, predictions);

		System.out.printf("        BCE =%.7f%n", BCE);
		System.out.println("        Confusion matrix");
		System.out.println("                        y=1      y=0");
		System.out.printf("                y^=1    %-8d %-8d%n", TP, FP);
		System.out.printf("                y^=0    %-8d %-8d%n", FN, TN);
		System.out.printf("        Accuracy =%.4f%n", accuracy);
		System.out.printf("        Precision =%.8f%n", precision);
		System.out.printf("        Recall =%.8f%n", recall);
		System.out.printf("        f1 score =%.8f%n", f1);
		System.out.printf("        auc roc =%.8f%n", aucRoc);

		return aucRoc;
	}

	private static void determineBestModel(Map<String, Double> metrics) {
		String bestModel = Collections.max(metrics.entrySet(), Map.Entry.comparingByValue()).getKey();
		System.out.printf("According to BCE, The best model is %s%n", bestModel);
		System.out.printf("According to Accuracy, The best model is %s%n", bestModel);
		System.out.printf("According to Precision, The best model is %s%n", bestModel);
		System.out.printf("According to Recall, The best model is %s%n", bestModel);
		System.out.printf("According to F1 score, The best model is %s%n", bestModel);
		System.out.printf("According to AUC ROC, The best model is %s%n", bestModel);
	}

	private static double computeAUC(List<Double> yTrue, List<Double> yPred) {
		int n = yTrue.size();
		int positives = (int) yTrue.stream().filter(y -> y == 1).count();
		int negatives = n - positives;
		List<Double> thresholds = new ArrayList<>();
		for (int i = 0; i <= 100; i++) thresholds.add(i / 100.0);

		List<Double> tpr = new ArrayList<>(), fpr = new ArrayList<>();
		for (double th : thresholds) {
			int TP = 0, FP = 0, FN = 0, TN = 0;
			for (int i = 0; i < n; i++) {
				int predBinary = (yPred.get(i) >= th) ? 1 : 0;
				if (yTrue.get(i) == 1) {
					if (predBinary == 1) TP++; else FN++;
				} else {
					if (predBinary == 1) FP++; else TN++;
				}
			}
			tpr.add(positives > 0 ? (double) TP / positives : 0);
			fpr.add(negatives > 0 ? (double) FP / negatives : 0);
		}

		double auc = 0.0;
		for (int i = 1; i < tpr.size(); i++) {
			auc += (tpr.get(i - 1) + tpr.get(i)) * Math.abs(fpr.get(i - 1) - fpr.get(i)) / 2;
		}
		return auc;
	}
}
