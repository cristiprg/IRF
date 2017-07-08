package org.apache.spark.ml.wahoo

import edu.mit.csail.db.ml.benchmarks.wahoo.WahooUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.mllib.tree.impl.TimeTracker
//import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{DataTypes, Metadata, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row


//import org.apache.spark.ml.classification.KNNClassifier
/**
  * Created by cristiprg on 20.05.17.
  */
object PointFeatures {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Wahoo")
      .setMaster("local[*]")
      .set("spark.driver.allowMultipleContexts", "true")

    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val sqlContext = new SQLContext(sc)

    //val trainingDataPath = "/media/cristiprg/Eu/YadoVR/pointFeatures.csv"
    val trainingDataPath = args(0)

    var df = loadPointFeaturesCSV(trainingDataPath, sqlContext)


    // transform the problem into a binary classification problem
    // Detect Ground only
    // https://stackoverflow.com/questions/30219592/create-new-column-with-function-in-spark-dataframe
    val coder = (trueLabel: String) => {if (trueLabel == "0") trueLabel else "1"}
    val sqlfunc = udf(coder)
    df = df.withColumn("binaryLabel", sqlfunc(col("label")))


    // Initialize pipeline elements
    val labelIndexer = new StringIndexer()
      .setInputCol("binaryLabel")
      //.setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)

    val featureAssembler = new VectorAssembler()
      //.setInputCols(Array("PF_Linearity", "PF_Planarity", "PF_Scattering"))
      .setInputCols(Array("PF_Linearity", "PF_Planarity", "PF_Scattering", "PF_Omnivariance", "PF_Eigenentropy", "PF_Anisotropy", "PF_CurvatureChange", "PF_AbsoluteHeight", "PF_LocalPointDensity3D", "PF_LocalRadius3D", "PF_MaxHeightDiff3D", "PF_HeightVar3D", "PF_LocalRadius2D", "PF_SumEigenvalues2D", "PF_RatioEigenvalues2D", "PF_MAccu2D", "PF_MaxHeightDiffAccu2D", "PF_VarianceAccu2D"))
      .setOutputCol("indexedFeatures")

    //val rf = new KNNClassifier()
    val rf = new org.apache.spark.ml.wahoo.WahooRandomForestClassifier()
    //val rf = new org.apache.spark.ml.classification.RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      //.setK(10)
      .setNumTrees(100)

    rf.regrowProp = 0.5
    rf.incrementalProp = 0.5

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
    //val Array(trainingData, trainingData2, testData) = df.randomSplit(Array(0.002, 0.698, 0.3))
    val (testData, trainingDataBatches) = splitDataInBatches(df, 0.3, 10)

    // Prepare test data
    val processedTestData = featureAssembler.transform(labelIndexer.transform(testData))


    // Initial training
    val timer = new TimeTracker()
    // train

    timer.start("training 0")
    var processedTrainingData = featureAssembler.transform(labelIndexer.transform(trainingDataBatches(0)))
    var model = rf.fit(processedTrainingData)
    var time = timer.stop("training 0")
    // predict
    var predictions = labelConverter.transform(
      model.transform(processedTestData)
    )
    // evaluate
    var accuracy = evaluator.evaluate(predictions)
    println("test error = " + (1.0 - accuracy))
    println("time = " + time)


    // Updates
    trainingDataBatches.drop(1).foreach(batch => {
      processedTrainingData = featureAssembler.transform(labelIndexer.transform(batch))
      timer.start("training " + batch)
      model = rf.update(model, processedTrainingData)
      // predict
      predictions = labelConverter.transform(
        model.transform(processedTestData)
      )
      time = timer.stop("training " + batch)
      // evaluate
      accuracy = evaluator.evaluate(predictions)
      println("test error = " + (1.0 - accuracy))
      println("time = " + time)
    })


    // Control benchmark
    println("Control benchmarks from here")
    val controlRF: org.apache.spark.ml.classification.RandomForestClassifier =
      new org.apache.spark.ml.classification.RandomForestClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("indexedFeatures")
        .setNumTrees(100)

    var currDF = sqlContext.createDataFrame(sc.emptyRDD[Row], df.schema)
    var controlModel: org.apache.spark.ml.classification.RandomForestClassificationModel = null
    trainingDataBatches.foreach(batch => {
      currDF = currDF.unionAll(batch)
      processedTrainingData = featureAssembler.transform(labelIndexer.transform(currDF))
      timer.start("training " + batch)
      controlModel = controlRF.fit(processedTrainingData)
      // predict
      predictions = labelConverter.transform(
        controlModel.transform(processedTestData)
      )
      time = timer.stop("training " + batch)
      // evaluate
      accuracy = evaluator.evaluate(predictions)
      println("test error = " + (1.0 - accuracy))
      println("time = " + time)
    })


  }

  /**
    * First batch is the test batch and the rest are training batches
    * @return
    */
  def splitDataInBatches(df: DataFrame, testPercentage: Double, nrTrainingBatches: Int) : (DataFrame, Array[DataFrame]) = {
    val Array(testData, trainingData) = df.randomSplit(Array(testPercentage, 1 - testPercentage))
    val weights = Array.fill[Double](nrTrainingBatches)(1 - testPercentage / nrTrainingBatches)

    // Here we assume the randomSplit normalizes the weights in order to sum up to 1.
    (testData, trainingData.randomSplit(weights))
  }

  def loadPointFeaturesCSV(filePath: String, sqlContext: SQLContext): DataFrame = {
    val customSchema = new StructType(Array[StructField](
      StructField("index", DataTypes.IntegerType, nullable = false, Metadata.empty),
      StructField("X", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("Y", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("Z", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("label", DataTypes.StringType, nullable = false, Metadata.empty),
      StructField("PF_Linearity", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_Planarity", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_Scattering", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_Omnivariance", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_Eigenentropy", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_Anisotropy", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_SumEigenvalues", DataTypes.DoubleType, nullable =  false, Metadata.empty),
      StructField("PF_CurvatureChange", DataTypes.DoubleType, nullable =  false, Metadata.empty),
      StructField("PF_AbsoluteHeight", DataTypes.DoubleType, nullable =  false, Metadata.empty),
      StructField("PF_LocalPointDensity3D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_LocalRadius3D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_MaxHeightDiff3D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_HeightVar3D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_LocalPointDensity2D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_LocalRadius2D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_SumEigenvalues2D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_RatioEigenvalues2D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_MAccu2D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_MaxHeightDiffAccu2D", DataTypes.DoubleType, nullable = false, Metadata.empty),
      StructField("PF_VarianceAccu2D", DataTypes.DoubleType, nullable = false, Metadata.empty))
    )

    WahooUtils.readData(filePath, sqlContext, customSchema)
  }
}
