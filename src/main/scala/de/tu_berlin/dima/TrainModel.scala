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
                 oldModel: RandomForestClassificationModel,
                 rf: org.apache.spark.ml.wahoo.WahooRandomForestClassifier) = {

    var model: RandomForestClassificationModel = oldModel


    for (i <- 0 until nrTiles) {
      val suffix = "part-" + "%05d".format(i)
      val tileFilePath = tilesDirectory + "/" + suffix
      logInfo("Processing tile: " + tileFilePath)

      logWarning("Extracting features for : " + tileFilePath)
      var df = PointFeaturesExtractor.extractPointFeatures(sc, tileFilePath,
        queryKNNScriptPath, buildKNNObjectScriptPath, buildPickles, kNNpicklePath + "-" + suffix)
      logWarning("Finished exracting features for : " + tileFilePath)

      val timer = new TimeTracker()
      timer.start("training 0")

      var processedTrainingData = featureAssembler.transform(df)

      if (model == null) {
        logInfo("Fitting IRF model!")
        logWarning("Fitting IRF model!")
        model = rf.fit(processedTrainingData)
        logWarning("Finished fitting IRF model!")
      }
      else {
        logInfo("Updating IRF model!")
        logWarning("Updating IRF model!")
        model = rf.update(model, processedTrainingData)
        logWarning("Finished updating IRF model!")
      }

      var time = timer.stop("training 0")
    }

    model
  }
}
