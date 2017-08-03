//package de.tu_berlin.dima
package org.apache.spark.ml.wahoo

import de.tu_berlin.dima.PointFeaturesExtractor
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.wahoo.PointFeatures.splitDataInBatches
import org.apache.spark.mllib.tree.impl.TimeTracker
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

import scala.io.Source

/**
  * Created by cristiprg on 03.08.17.
  */
object TrainModel extends Logging{

  def main(args:Array[String]) = {

    // Parse args
    val laserPointsFile = args(0)
    val batchSize = args(1).toInt
    val queryKNNScriptPath = args(2)
    val buildKNNObjectScriptPath = args(3)
    val buildPickles = if(args(4).toInt == 0) false else true
    val kNNpicklePath = args(5)
    val outputFile = args(6)

    val tilesDirectory = laserPointsFile + "_tiles_folder"

    val conf = new SparkConf()
      .setAppName("Wahoo")
      .set("spark.driver.allowMultipleContexts", "true")

    val sc = new SparkContext(conf)
    val MAX_NUMBER_OF_TILES: Int = 100
    val laserPointsRDD = sc.textFile(laserPointsFile, MAX_NUMBER_OF_TILES)
    val N = laserPointsRDD.count() // Get the number of points
    val T = batchSize // Number of points in every tile
    val nrTiles = Math.ceil(N.toDouble/T).toInt// Number of tiles, i.e. the number of paritions on this particular RDD

    if (nrTiles > MAX_NUMBER_OF_TILES) {
      logError("The resulting number of tiles ("+ nrTiles +") is bigger than the maximum allowed ("+MAX_NUMBER_OF_TILES+"), please increase the batch size!")
      sc.stop()
      sys.exit(1)
    }

    laserPointsRDD.coalesce(nrTiles).saveAsTextFile(tilesDirectory) // Save tiles as files, then process them one by one

    val rf = new org.apache.spark.ml.wahoo.WahooRandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(100)

    var model: RandomForestClassificationModel = null

    for (i <- 0 until nrTiles) {
      val suffix = "part-" + "%05d".format(i)
      val tileFilePath = tilesDirectory + "/" + suffix
      logInfo("Processing tile: " + tileFilePath)

      var df = PointFeaturesExtractor.extractPointFeatures(sc, tileFilePath,
        queryKNNScriptPath, buildKNNObjectScriptPath, buildPickles, kNNpicklePath + "-" + suffix)

      // Initialize pipeline elements
      val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(df)

      val featureAssembler = new VectorAssembler()
//        .setInputCols(Array("PF_Linearity", "PF_Planarity", "PF_Scattering", "PF_Omnivariance", "PF_Eigenentropy", "PF_Anisotropy", "PF_CurvatureChange", "PF_AbsoluteHeight", "PF_LocalPointDensity3D", "PF_LocalRadius3D", "PF_MaxHeightDiff3D", "PF_HeightVar3D", "PF_LocalRadius2D", "PF_SumEigenvalues2D", "PF_RatioEigenvalues2D", "PF_MAccu2D", "PF_MaxHeightDiffAccu2D", "PF_VarianceAccu2D"))
        .setInputCols(Array("PF_AbsoluteHeight", "PF_Linearity", "PF_Planarity","PF_Scattering", "PF_Omnivariance", "PF_Anisotropy", "PF_Eigenentropy", "PF_CurvatureChange"))
        .setOutputCol("indexedFeatures")



      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)

      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("precision")


      // https://spark.apache.org/docs/1.5.2/ml-ensembles.html#random-forests

      // Split the data into training and test sets (30% held out for testing)
//      val Array(trainingData, trainingData2, testData) = df.randomSplit(Array(0.002, 0.698, 0.3))
//      val (testData, trainingDataBatches) = splitDataInBatches(df, fractionTest, numBatches)
      val Array(testData, trainingData) = df.randomSplit(Array(0.25, 0.75))
      println("testData.count() = " + testData.count())

      // Prepare test data
      val processedTestData = featureAssembler.transform(labelIndexer.transform(testData))


      // Initial training
      val timer = new TimeTracker()
      // train

      timer.start("training 0")
      var processedTrainingData = featureAssembler.transform(labelIndexer.transform(trainingData))
      println("Fitting, trainingDataBatches(0).count() = " + trainingData.count())

      if (i == 0) {
        logInfo("Fitting IRF model!")
        model = rf.fit(processedTrainingData)
      }
      else {
        logInfo("Updating IRF model!")
        model = rf.update(model, processedTrainingData)
      }

      var time = timer.stop("training 0")
      // predict
      var predictions = labelConverter.transform(
        model.transform(processedTestData)
      )
      // evaluate
      println("Evaluating ...")
      var accuracy = evaluator.evaluate(predictions)

      println("test error = " + (1.0 - accuracy))
      println("time = " + time)


    }

//    //sc.setLogLevel("WARN")
//    val sqlContext = new SQLContext(sc)
//
//    var df = PointFeaturesExtractor.extractPointFeatures(sc, trainingDataPath,
//      "/media/cristiprg/Eu/Scoala/BDAPRO/LidarPointFeaturesScala/src/main/scripts/queryKNN.py",
//      "/media/cristiprg/Eu/Scoala/BDAPRO/LidarPointFeaturesScala/src/main/scripts/buildKNNObject.py",
//      buildPickle = true,
//      "/media/cristiprg/Eu/Scoala/BDAPRO/LidarPointFeaturesScala/knnObj.pkl")


  }

}
