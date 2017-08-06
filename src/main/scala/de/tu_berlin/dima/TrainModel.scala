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

  def trainModel(sc: SparkContext,
                 featureAssembler: VectorAssembler,
                 tilesDirectory: String,
                 nrTiles: Int,
                 queryKNNScriptPath: String,
                 buildKNNObjectScriptPath: String,
                 buildPickles: Boolean,
                 kNNpicklePath: String,
                 savedModelPath: String) = {


    val rf = new org.apache.spark.ml.wahoo.WahooRandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(100)

    var model: RandomForestClassificationModel = null


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

    // save model
    sc.parallelize(Seq(model), 1).saveAsObjectFile(savedModelPath)
  }
}
