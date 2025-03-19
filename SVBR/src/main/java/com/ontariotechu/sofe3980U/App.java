package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.*;
import com.opencsv.*;

/**
 * Evaluate Single Variable Binary Regression
 */
public class App {
    public static void main(String[] args) {
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

        System.out.printf("Best performing model: %s with AUC-ROC = %.6f%n", bestModel, bestAUC);
    }

    private static double evaluateModel(String filePath) {
        List<String[]> allData;
        try (FileReader filereader = new FileReader(filePath);
             CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build()) {
            allData = csvReader.readAll();
        } catch (Exception e) {
            System.err.println("Error reading the CSV file: " + filePath);
            return 0.0;
        }

        int TP = 0, FP = 0, TN = 0, FN = 0;
        double BCE = 0.0;
        int n = allData.size();
        double threshold = 0.5;
        List<Double> yTrueList = new ArrayList<>(), yPredList = new ArrayList<>();

        for (String[] row : allData) {
            double yTrue = Double.parseDouble(row[0]);
            double yPred = Double.parseDouble(row[1]);
            int yPredBinary = (yPred >= threshold) ? 1 : 0;

            if (yTrue == 1) {
                if (yPredBinary == 1) TP++; else FN++;
            } else {
                if (yPredBinary == 1) FP++; else TN++;
            }

            BCE += yTrue * Math.log(yPred + 1e-9) + (1 - yTrue) * Math.log(1 - yPred + 1e-9);
            yTrueList.add(yTrue);
            yPredList.add(yPred);
        }

        BCE = -BCE / n;
        double accuracy = (double) (TP + TN) / n;
        double precision = (TP + FP) > 0 ? (double) TP / (TP + FP) : 0;
        double recall = (TP + FN) > 0 ? (double) TP / (TP + FN) : 0;
        double f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        double aucRoc = calculateAUC(yTrueList, yPredList);

        System.out.printf("BCE: %.7f%n", BCE);
        System.out.printf("Confusion Matrix: TP=%d FP=%d TN=%d FN=%d%n", TP, FP, TN, FN);
        System.out.printf("Accuracy: %.4f%nPrecision: %.8f%nRecall: %.8f%nF1 Score: %.8f%nAUC-ROC: %.8f%n",
                accuracy, precision, recall, f1Score, aucRoc);

        return aucRoc;
    }

    private static double calculateAUC(List<Double> yTrue, List<Double> yPred) {
        int n = yTrue.size();
        int nPositive = (int) yTrue.stream().filter(y -> y == 1).count();
        int nNegative = n - nPositive;
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
            tpr.add(nPositive > 0 ? (double) TP / nPositive : 0);
            fpr.add(nNegative > 0 ? (double) FP / nNegative : 0);
        }

        double auc = 0.0;
        for (int i = 1; i < tpr.size(); i++) {
            auc += (tpr.get(i - 1) + tpr.get(i)) * Math.abs(fpr.get(i - 1) - fpr.get(i)) / 2;
        }
        return auc;
    }
}
