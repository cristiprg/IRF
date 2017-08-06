package de.tu_berlin.dima

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.apache.spark.ml.wahoo.TrainModel
import org.apache.spark.ml.wahoo.TrainModel.logError

/**
  * Created by cristiprg on 06.08.17.
  */
object IRF extends Logging{

  def splitFilesIntoTiles(sc: SparkContext, laserPointsFile: String, tilesDirectory: String, batchSize: Int): Int = {
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
    nrTiles
  }

  //TODO: 1) toate pointfeature-urile calculate si dupaia batchuri IRF (ca inainte defapt, nu stiu ce plm te-ai complicat)
  def main(args: Array[String]) = {
    // Parse args
    val command: String = args(0)
    val laserPointsFile = args(1)
    val batchSize = args(2).toInt
    val queryKNNScriptPath = args(3)
    val buildKNNObjectScriptPath = args(4)
    val buildPickles = if(args(5).toInt == 0) false else true
    val kNNpicklePath = args(6)
    val savedModelPath = args(7)
    val outputFile = args(8)

    command match {
      case "train" | "predict" =>

        val conf = new SparkConf()
          .setAppName("Wahoo")
          .set("spark.driver.allowMultipleContexts", "true")
        val sc = new SparkContext(conf)

        val tilesDirectory = laserPointsFile + "_tiles_folder" // @temporary
        val nrTiles = splitFilesIntoTiles(sc, laserPointsFile, tilesDirectory, batchSize)

        // Declare here in order to avoid code duplicate in train+predict
        val featureAssembler = new VectorAssembler()
          //        .setInputCols(Array("PF_Linearity", "PF_Planarity", "PF_Scattering", "PF_Omnivariance", "PF_Eigenentropy", "PF_Anisotropy", "PF_CurvatureChange", "PF_AbsoluteHeight", "PF_LocalPointDensity3D", "PF_LocalRadius3D", "PF_MaxHeightDiff3D", "PF_HeightVar3D", "PF_LocalRadius2D", "PF_SumEigenvalues2D", "PF_RatioEigenvalues2D", "PF_MAccu2D", "PF_MaxHeightDiffAccu2D", "PF_VarianceAccu2D"))
          .setInputCols(Array("PF_AbsoluteHeight", "PF_Linearity", "PF_Planarity","PF_Scattering", "PF_Omnivariance", "PF_Anisotropy", "PF_Eigenentropy", "PF_CurvatureChange"))
          .setOutputCol("indexedFeatures")

        command match {
          case "train" =>
            TrainModel.trainModel(sc, featureAssembler, tilesDirectory, nrTiles, queryKNNScriptPath,
              buildKNNObjectScriptPath, buildPickles, kNNpicklePath, savedModelPath)
          case "predict" =>
            Predict.predict(sc, featureAssembler, tilesDirectory, nrTiles, queryKNNScriptPath,
              buildKNNObjectScriptPath, buildPickles, kNNpicklePath, savedModelPath, outputFile)
        }

      case _ =>
        println("ERROR: command must be either \"train\" or \"predict\"!")
        sys.exit(-1)
    }
  }
}
