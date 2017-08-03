#!/bin/bash

# Don't forget to install/bootstrap AWS EMR with Python packages
# pip install --user scikit-learn
# pip install --user pandas
# pip install --user s3fs


echo "Don't forget to set \
spark.executor.memory            20000M \
spark.executor.cores             8 \
spark.default.parallelism        100 \
spark.executor.instances         5 \
spark.driver.extraJavaOptions    -Xmx10240m \
"

# Check input args in file NearestNeighborsPython.scala
#laserPointsFile="/media/cristiprg/Eu/YadoVR/binaries/data/semantic3d/bildstein_station1_xyz_intensity_rgb_1K.csv"
laserPointsFile="s3://spark-pointfeatures/Enschede_1_test_space.csv"
batchSize=5000
queryKNNScriptPath="/home/hadoop/queryKNN.py"
buildKNNObjectScriptPath="/home/hadoop/buildKNNObject.py"
buildPickles="1"
kNNpicklePath="/home/hadoop/knnObj.pkl"
outputFile="s3://spark-pointfeatures/pointFeatures.csv"

# Remove existing hadoop folder
# /usr/local/hadoop/bin/hdfs dfs -rm -r /Spark/data/pointFeatures.csv
aws s3 rm --recursive $outputFile
aws s3 rm --recursive ${laserPointsFile}_tiles_folder

# Run application
#SPARK_FOLDER="/home/cristiprg/Programs/spark-2.1.1-bin-hadoop2.7"
#SPARK_FOLDER="/home/cristiprg/Programs/spark-1.6.3-bin-hadoop2.6"

spark-submit --master yarn --class org.apache.spark.ml.wahoo.TrainModel \
        /home/hadoop/ml.jar \
        $laserPointsFile $batchSize $queryKNNScriptPath $buildKNNObjectScriptPath $buildPickles \
        $kNNpicklePath $outputFile
