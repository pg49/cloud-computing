package org.project;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class SparkProject {
    public static void main(String[] args) {
        // Validate command line arguments
        if (args.length != 3) {
            System.err.println("Usage: SparkProject <training-data-path> <test-data-path> <output-path>");
            System.exit(1);
        }

        String trainingDataPath = args[0];
        String testDataPath = args[1];
        String outputPath = args[2] + "model";

        // Create a Spark session
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPrediction")
                .config("spark.master", "local")
                .getOrCreate();

        // Load training and testing data
        Dataset<Row> trainingData = loadData(spark, trainingDataPath);
        Dataset<Row> testData = loadData(spark, testDataPath);

        // Prepare data
        prepareData(trainingData);
        prepareData(testData);

    }
}