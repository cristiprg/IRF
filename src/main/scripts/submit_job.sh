#!/bin/bash

# Check input args in file NearestNeighborsPython.scala
laserPointsFile="/media/cristiprg/Eu/YadoVR/binaries/data/semantic3d/bildstein_station1_xyz_intensity_rgb_1K.csv"
queryKNNScriptPath="/media/cristiprg/Eu/Scoala/BDAPRO/LidarPointFeaturesScala/src/main/scripts/queryKNN.py"
buildKNNObjectScriptPath="/media/cristiprg/Eu/Scoala/BDAPRO/LidarPointFeaturesScala/src/main/scripts/buildKNNObject.py"
buildPickles="1"
kNNpicklePath="/media/cristiprg/Eu/Scoala/BDAPRO/LidarPointFeaturesScala/knnObj.pkl"
outputFile="hdfs://localhost:54310/Spark/data/pointFeatures.csv"

# Remove existing hadoop folder
/usr/local/hadoop/bin/hdfs dfs -rm -r /Spark/data/pointFeatures.csv

# Run application
#SPARK_FOLDER="/home/cristiprg/Programs/spark-2.1.1-bin-hadoop2.7"
SPARK_FOLDER="/home/cristiprg/Programs/spark-1.6.3-bin-hadoop2.6"

$SPARK_FOLDER/bin/spark-submit --master local --class de.tu_berlin.dima.NearestNeighborsPython \
	/media/cristiprg/Eu/Scoala/BDAPRO/LidarPointFeaturesScala/target/scala-2.10/LidarPointFeaturesSpark.jar \
	$laserPointsFile $queryKNNScriptPath $buildKNNObjectScriptPath $buildPickles \
	$kNNpicklePath $outputFile
