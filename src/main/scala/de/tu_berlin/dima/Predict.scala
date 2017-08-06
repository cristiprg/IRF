package de.tu_berlin.dima

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.wahoo.RandomForestClassificationModel
import org.apache.spark.{Logging, SparkContext}
import org.apache.spark.ml.wahoo.TrainModel.{logError, logInfo}
import org.apache.spark.sql.{SQLContext, SaveMode}
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
              outputFile: String) = {



    val classifiedPointsParquetFile = "classifedPoints.parquet" // @temporary

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
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("quoteMode", "NONE")
      .option("escape","\\")
      .save(outputFile)

//    delete temporary file
    "hdfs dfs -rm -r " + classifiedPointsParquetFile !
  }
}
