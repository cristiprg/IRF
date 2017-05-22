package org.apache.spark.ml.wahoo

import edu.mit.csail.db.ml.benchmarks.wahoo.WahooUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
//import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{DataTypes, Metadata, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}

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
    val sqlContext = new SQLContext(sc)

    val trainingDataPath = "/media/cristiprg/Eu/YadoVR/pointFeatures.csv"

    val df = loadPointFeaturesCSV(trainingDataPath, sqlContext)

    df.printSchema()
    df.show()

    // https://spark.apache.org/docs/1.5.2/ml-ensembles.html#random-forests

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))


    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)

    val featureAssembler = new VectorAssembler()
      //.setInputCols(Array("PF_Linearity", "PF_Planarity", "PF_Scattering"))
      .setInputCols(Array("PF_Linearity", "PF_Planarity", "PF_Scattering", "PF_Omnivariance", "PF_Eigenentropy", "PF_Anisotropy", "PF_CurvatureChange", "PF_AbsoluteHeight", "PF_LocalPointDensity3D", "PF_LocalRadius3D", "PF_MaxHeightDiff3D", "PF_HeightVar3D", "PF_LocalRadius2D", "PF_SumEigenvalues2D", "PF_RatioEigenvalues2D", "PF_MAccu2D", "PF_MaxHeightDiffAccu2D", "PF_VarianceAccu2D"))
      .setOutputCol("indexedFeatures")

    val rf = new org.apache.spark.ml.classification.RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(100)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureAssembler, rf, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)
    predictions.select("predictedLabel", "label", "indexedFeatures").show(5)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val accuracy = evaluator.evaluate(predictions)
    println("test error = " + (1.0 - accuracy))

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
      StructField("PF_VarianceAccu2D", DataTypes.DoubleType, nullable = false, Metadata.empty)))

    WahooUtils.readData(filePath, sqlContext, customSchema)
  }
}
