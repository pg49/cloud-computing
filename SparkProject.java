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
    public static void main(String[] args){
        if (args.length != 3) {
            System.err.println("Usage: SparkProject <training-data-path> <test-data-path> <output-path>");
            System.exit(1);
        }

        String trainingData = args[0];
        String testData = args[1];
        String output = args[2] + "test.model";

        SparkSession spark = SparkSession.builder()
                .appName("SparkProject")
                .config("spark.master", "local")
                .getOrCreate();

        Dataset<Row> trainingData = loadData(spark, trainingData);
        Dataset<Row> testData = loadData(spark, testData);

        prepareData(trainingData);
        prepareData(testData);

        String[] inputColumns = {"fixed acidity", "volatile acidity", "citric acid", "chlorides", "total sulfur dioxide", "density", "sulphates", "alcohol"};

        VectorAssembler assembler = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features");

        RandomForestClassifier rfc = new RandomForestClassifier().setLabelCol("quality").setFeaturesCol("features").setImpurity("entropy");

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rfc});

        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);

        evaluateModel(predictions);

        model.write().overwrite().save(output);
    }

    private static Dataset<Row> loadData(SparkSession spark, String path) {
        return spark.read().format("csv").option("header", true).option("quote", "\"").option("delimiter", ";").load(path);
    }

    private static void prepareData(Dataset<Row> data) {
        String[] inputColumns = {"fixed acidity", "volatile acidity", "citric acid", "chlorides", "total sulfur dioxide", "density", "sulphates", "alcohol"};
        for (String col : inputColumns) {
            data = data.withColumn(col, data.col(col).cast("Double"));
        }

        data = data.withColumn("quality", data.col("quality").cast("Double"));
        data = data.withColumn("label", functions.when(data.col("quality").geq(7), 1.0).otherwise(0.0));
    }

    private static void evaluateModel(Dataset<Row> predictions) {
        MulticlassClassificationEvaluator modelEval = new MulticlassClassificationEvaluator()
        .setLabelCol("quality")
        .setPredictionCol("prediction");

        double accuracy = modelEval.setMetricName("accuracy").evaluate(predictions);
        double f1Score = modelEval.setMetricName("f1").evaluate(predictions);

        System.out.println("F1 score is " + f1Score);
        System.out.println("Accuracy is " + accuracy);
    }
}