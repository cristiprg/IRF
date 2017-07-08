package de.tu_berlin.dima

import breeze.linalg.DenseVector
import org.apache.spark.{SparkContext, SparkFiles}
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.mutable.ArrayBuffer
import sys.process._


/**
  * Helper class that extracts the laser point features defined by Weinmann et. al 2014 from a csv file that contains
  * the coordinates of the points. Unfortunately, for now it has external dependencies two python scripts which need
  * to be located on the disk, accessible by JVM. This is necessary since there is no good kNN in Spark yet, so we
  * use sklearn's Nearest Neighbors.
  */
object PointFeaturesExtractor {


  /**
    * Extracts the point features from a csv file of this format: x,y,z,c.
    * @param laserPointsFile Local path to csv file with laser points. No HDFS supported.
    * @param queryKNNScriptPath Local path to python query script.
    * @param buildKNNObjectScriptPath Local path to python script that builds the kNN object.
    * @param buildPickle Whether or not to rebuild the kNN object (takes a while)
    * @param kNNpicklePath Local path to the pickled (serialized) kNN object
    * @return a DataFrame of this form: index,x,y,c,f1,f2,...,fn,c
    */
  def extractPointFeatures(
                            sc: SparkContext,
                            laserPointsFile: String,
                            queryKNNScriptPath: String,
                            buildKNNObjectScriptPath: String,
                            buildPickle: Boolean,
                            kNNpicklePath: String): DataFrame = {

    val queryKNNScriptName = queryKNNScriptPath.split("/").last
    val kNNpickleName = kNNpicklePath.split("/").last

    val csvSeparator = " "

    // Build lookup table for quickly retrieving the coordinates of the point at certain index in the original file.
    // Performing this in Python is for some reason extremely slow, so we're doing this in Scala. Therefore LUT will
    // be broadcast to all workers.
    val LUT = buildLUT(laserPointsFile, csvSeparator)

    // Step 1: compute NN, i.e. run python script that builds a sklearn.neighbors.NearestNeighbors object and then
    // saves it to disk (pickles it).
    val pythonBuildKNNCMD = "python " + buildKNNObjectScriptPath + " " + laserPointsFile + " " + kNNpicklePath
    if (buildPickle)
      issueBuildKNNCMD(pythonBuildKNNCMD)


    // Some householding: make an RDD out of the input file and copy necessary files for querying to workers.
    sc.addFile(queryKNNScriptPath)
    sc.addFile(kNNpicklePath)

    // Read laser points
    val laserPointsFileRDD = sc.textFile(laserPointsFile, 3)
    laserPointsFileRDD.cache()

    // Get the x, y, z coordinates of the points in one RDD and the labels in another RDD. This helps reduce the amount
    // sent via pipe to the python query script. Ultimately, these two are going to be merged (joined) back together.
    val laserPointsRDD = laserPointsFileRDD
      .map(x => {
        val splits = x.split(csvSeparator)
        splits(0) + "," + splits(1) + "," + splits(2)
      })

    // Get the labels of the points
    val laserPointsLabelsRDD = laserPointsFileRDD
      .map(_.split(csvSeparator)(3)) // fourth position in the csv file is the label
      .zipWithIndex()
      .map(x => (x._2, x._1)) // put the key on the first position (for join operation)


    // Step 2. retrieve NN and compute the features. Run the python query script and send the query points via stdin.
    // This is similar to a RDD.map() function which takes an external command as argument, instead of a function.
    // The script outputs on each line the list the ids in the original dataset of the nearest neighbors of each query
    // point. The ids are then looked up to get the real coordinates and then the features are computed.
    // Note: the first element of each line is the id of the query point.
    val pythonQueryKNNCMD = "python " + SparkFiles.get(queryKNNScriptName) + " " + SparkFiles.get(kNNpickleName) // args to script are locations of the pkls
    val laserPointFeaturesRDD = laserPointsRDD
      .pipe(pythonQueryKNNCMD)
      .zipWithIndex() // perform this as early as possible, it saved 33% of execution time
      .map(x => (x._2, x._1)) // put the key on the first position (for join operation)
      .map(x => (x._1, x._2.split(',').map(x => LUT(x.trim.toInt)))) // TODO: check whether the lookup operation is very slow here
      //    .map(_.toList.map(_.toList)) // useful for debugging
      .map(x => (x._1, prepareDataFrame(x._2)))


    // "Attach" the label column by joining on the same index.
    val finalRDD = laserPointFeaturesRDD
      .join(laserPointsLabelsRDD)
      .map(x => (x._1, x._2._1._1, x._2._1._2, x._2._1._3, x._2._1._4, x._2._1._5, x._2._1._6, x._2._1._7, x._2._1._8,
        x._2._1._9, x._2._1._10, x._2._2)) // flatten the tuple

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val laserPointsFeaturesDF = finalRDD.toDF("index", "X", "Y", "Z", "PF_Linearity", "PF_Planarity",
      "PF_Scattering", "PF_Omnivariance", "PF_Anisotropy", "PF_Eigenentropy", "PF_CurvatureChange", "label")

    laserPointsFeaturesDF
  }


  /**
    * Reads the file and builds an Array that contains the coordinates of the laser points. This is useful when we need
    * the coordinates of the point at a certain index in the file.
    */
  private def buildLUT(laserPointsFile: String, csvSeparator: String): Array[Array[Double]] = {
    // http://alvinalexander.com/scala/csv-file-how-to-process-open-read-parse-in-scala
    // each row is an array of double (the columns in the csv file)
    val rows = ArrayBuffer[Array[Double]]()

    // (1) read the csv data
    using(scala.io.Source.fromFile(laserPointsFile)) { source =>
      for (line <- source.getLines) {
        // keep the first three columns, i.e. the coordinates
        rows += line.split(csvSeparator).slice(0, 3).map(_.trim.toDouble)
      }
    }

    def using[A <: { def close(): Unit }, B](resource: A)(f: A => B): B =
      try {
        f(resource)
      } finally {
        resource.close()
      }

    rows.toArray
  }

  /**
    * Computes the features of the query point (which is at index 0) and returns them in a friendly manner to processed
    * further in a dataframe.
    * @param neighbors the nearest neighbors of the query point. The query point is found in this variable at index 0
    * @return tuple of this form: x, y, z, feature 1, feature 2, ..., feature n
    */
  private def prepareDataFrame(neighbors: Array[Array[Double]]) = {
    val queryPoint = neighbors(0)
    val features = getPointFeatures(neighbors)

    (queryPoint(0), queryPoint(1), queryPoint(2), features._1, features._2, features._3, features._4, features._5, features._6, features._7)
  }

  private def getPointFeatures(neighbors: Array[Array[Double]]) = {
    val (optNeighSize, optEntropy, optEignValues) = getOptimalNeighSize(neighbors)

    val e1 = optEignValues(0)
    val e2 = optEignValues(1)
    val e3 = optEignValues(2)

    val linearity = (e1-e2)/e1
    val planarity = (e2-e3)/e1
    val scattering = e3/e1
    val omnivariance = Math.pow(e1*e2*e3, 0.333)
    val anisotropy = (e1-e3)/e1
    val eigenentropy = optEntropy
    val changeOfCurvature = e3 / (e1+e2+e3)

    (linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, changeOfCurvature)
  }


  /**
    * Computes the optimal neigh size by brute-force, i.e. trying (almost) all the values within certain range. The size
    * for which the entropy is minimum is selected according to Weinmann et. al ch. 2.1 (end of the chapter).
    */
  private def getOptimalNeighSize(neighbors: Array[Array[Double]]) = {
    // TODO: set the search parameters, i.e. min and step, as application top-level parameters
    val minNeighSize = 10
    val maxNeighSize = neighbors.length
    val stepNeighSize = 5
    val nrDim = 3
    var optEntropy = Double.MaxValue
    var optNeighSize = 0
    var optEignValues: DenseVector[Double] = null

    for (currNeighSize <- minNeighSize to maxNeighSize by stepNeighSize) {
      val neighborsSubset = neighbors.slice(0, currNeighSize) // TODO: this might be slow, consider making use of the offset param below
      val breezeMatrix = new breeze.linalg.DenseMatrix(neighborsSubset.length, nrDim, neighborsSubset.flatten, 0, nrDim, true)
      val eignValues = breeze.linalg.princomp(breezeMatrix).propvar
      val entropy = -eignValues(0) * scala.math.log(eignValues(0)) - eignValues(1) * scala.math.log(eignValues(1)) - eignValues(2) * scala.math.log(eignValues(2))

      if (entropy < optEntropy) {
        optEntropy = entropy
        optNeighSize = currNeighSize
        optEignValues = eignValues
      }

    }
    (optNeighSize, optEntropy, optEignValues)
  }

  private def issueBuildKNNCMD(cmd: String) = {
    println("Building KNN Object, have little patience please, this step is not distributed ...")

    val ret_code = cmd !

    if (ret_code == 0)
      println("Done building KNN Object!")
    else {
      println("Error building KNN Object!")
      System.exit(ret_code)
    }


  }
}
