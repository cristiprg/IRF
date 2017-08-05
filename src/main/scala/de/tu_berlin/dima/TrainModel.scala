//package de.tu_berlin.dima
package org.apache.spark.ml.wahoo

import de.tu_berlin.dima.PointFeaturesExtractor
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.wahoo.PointFeatures.splitDataInBatches
import org.apache.spark.mllib.tree.impl.TimeTracker
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SaveMode}

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
    val classifiedPointsParquetFile = "classifedPoints.parquet"

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
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(100)

    var model: RandomForestClassificationModel = null

    val featureAssembler = new VectorAssembler()
      //        .setInputCols(Array("PF_Linearity", "PF_Planarity", "PF_Scattering", "PF_Omnivariance", "PF_Eigenentropy", "PF_Anisotropy", "PF_CurvatureChange", "PF_AbsoluteHeight", "PF_LocalPointDensity3D", "PF_LocalRadius3D", "PF_MaxHeightDiff3D", "PF_HeightVar3D", "PF_LocalRadius2D", "PF_SumEigenvalues2D", "PF_RatioEigenvalues2D", "PF_MAccu2D", "PF_MaxHeightDiffAccu2D", "PF_VarianceAccu2D"))
      .setInputCols(Array("PF_AbsoluteHeight", "PF_Linearity", "PF_Planarity","PF_Scattering", "PF_Omnivariance", "PF_Anisotropy", "PF_Eigenentropy", "PF_CurvatureChange"))
      .setOutputCol("indexedFeatures")

    for (i <- 0 until nrTiles) {
      val suffix = "part-" + "%05d".format(i)
      val tileFilePath = tilesDirectory + "/" + suffix
      logInfo("Processing tile: " + tileFilePath)

      var df = PointFeaturesExtractor.extractPointFeatures(sc, tileFilePath,
        queryKNNScriptPath, buildKNNObjectScriptPath, buildPickles, kNNpicklePath + "-" + suffix)

      val timer = new TimeTracker()
      timer.start("training 0")

      var processedTrainingData = featureAssembler.transform(df)

      if (i == 0) {
        logInfo("Fitting IRF model!")
        model = rf.fit(processedTrainingData)
      }
      else {
        logInfo("Updating IRF model!")
        model = rf.update(model, processedTrainingData)
      }

      var time = timer.stop("training 0")
    }

    for (i <- 0 until nrTiles) {
      val suffix = "part-" + "%05d".format(i)
      val tileFilePath = tilesDirectory + "/" + suffix
      logInfo("Processing tile: " + tileFilePath)

      var df = PointFeaturesExtractor.extractPointFeatures(sc, tileFilePath,
        queryKNNScriptPath, buildKNNObjectScriptPath, buildPickles, kNNpicklePath + "-" + suffix)

      val processedTestData = featureAssembler.transform(df)
      var predictions = model.transform(processedTestData)
      println("Exporting predicitons to parquet file ...")
      if (i == 0) {
        //        create table
        predictions.select("X", "Y", "Z", "prediction").write.parquet(classifiedPointsParquetFile)
      }
      else{
        //        append to table
        predictions.select("X", "Y", "Z", "prediction").write.mode(SaveMode.Append).parquet(classifiedPointsParquetFile)
      }
    }

    val sqlContext = new SQLContext(sc)
    sqlContext
      .read.parquet(classifiedPointsParquetFile)
      .coalesce(1) // reduce to one partition
      .write.format("com.databricks.spark.csv").option("header", "true").save(outputFile + "-classified")

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
