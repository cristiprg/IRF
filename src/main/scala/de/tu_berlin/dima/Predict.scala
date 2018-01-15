package de.tu_berlin.dima

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.wahoo.RandomForestClassificationModel
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.ml.wahoo.TrainModel.{logError, logInfo}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode}

import sys.process._

/**
  * Created by cristiprg on 06.08.17.
  */
object Predict extends Logging{

  def predict(sc: SparkContext,
              featureAssembler: VectorAssembler,
              tilesDirectory: String,
              nrTiles: Int,
              queryKNNScriptPath: String,
              buildKNNObjectScriptPath: String,
              buildPickles: Boolean,
              kNNpicklePath: String,
              model: RandomForestClassificationModel,
              outputFile: String,
              doEvaluation: Boolean) = {



    val classifiedPointsParquetFile = "classifedPoints.parquet" // @temporary
    var globalPredictionAndLabels: RDD[(Double, Double)] = sc.emptyRDD[(Double, Double)]// for evluating everything at once, instead of per tile

    for (i <- 0 until nrTiles) {
      val suffix = "part-" + "%05d".format(i)
      val tileFilePath = tilesDirectory + "/" + suffix
      logInfo("Processing tile: " + tileFilePath)

      logWarning("Extracting features for : " + tileFilePath)
      var df = PointFeaturesExtractor.extractPointFeatures(sc, tileFilePath,
        queryKNNScriptPath, buildKNNObjectScriptPath, buildPickles, kNNpicklePath + "-" + suffix)
      logWarning("Finished exracting features for : " + tileFilePath)

      val processedTestData = featureAssembler.transform(df)

      logWarning("Predicting!")
      var predictions = model.transform(processedTestData)
      logWarning("Finished predicting!")

      println("Exporting predicitons to parquet file ...")
      if (i == 0) {
        //        create table
        predictions.select("X", "Y", "Z", "prediction").write.parquet(classifiedPointsParquetFile)
      }
      else{
        //        append to table
        predictions.select("X", "Y", "Z", "prediction").write.mode(SaveMode.Append).parquet(classifiedPointsParquetFile)
      }

      if (doEvaluation) {
        //        performEvaluation(predictions)
        val predictionAndLabels = predictions.select("prediction", "label").rdd.map(
          x => (x.get(0).asInstanceOf[Double], x.get(1).asInstanceOf[Double])
        )

        globalPredictionAndLabels = globalPredictionAndLabels.union(predictionAndLabels)
      }

    }

    if (doEvaluation) printEvalMetrics(globalPredictionAndLabels)

    val sqlContext = new SQLContext(sc)
    sqlContext
      .read.parquet(classifiedPointsParquetFile)
      .coalesce(1) // reduce to one partition
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("quoteMode", "NONE")
      .option("escape","\\")
      .save(outputFile)

//    delete temporary file
    "hdfs dfs -rm -r " + classifiedPointsParquetFile !
  }

  private def printEvalMetrics(predictionAndLabels: RDD[(Double, Double)]) : Unit = {
    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val precision = metrics.precision
    val recall = metrics.recall // same as true positive rate
    val f1Score = metrics.fMeasure
    println("Summary Statistics")
    println(s"Precision = $precision")
    println(s"Recall = $recall")
    println(s"F1 Score = $f1Score")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }

  private def performEvaluation(predictions: DataFrame) : Unit = {

//    https://spark.apache.org/docs/1.6.3/mllib-evaluation-metrics.html
    println("Evaluating ...")
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("precision")

    // Get raw scores on the test set (and convert to RDD because MultiClassMetrics doesn't support dataframes
    val predictionAndLabels = predictions.select("prediction", "label").rdd.map(
      x => (x.get(0).asInstanceOf[Double], x.get(1).asInstanceOf[Double])
    )

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val precision = metrics.precision
    val recall = metrics.recall // same as true positive rate
    val f1Score = metrics.fMeasure
    println("Summary Statistics")
    println(s"Precision = $precision")
    println(s"Recall = $recall")
    println(s"F1 Score = $f1Score")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

//    evaluator.evaluate(predictions)
  }

}
