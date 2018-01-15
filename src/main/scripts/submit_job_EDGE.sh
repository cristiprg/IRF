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

SPARK_HOME='/app/spark-1.6.3-bin-hadoop2.6/bin'

# Check input args in file NearestNeighborsPython.scala
#laserPointsFile="/media/cristiprg/Eu/YadoVR/binaries/data/semantic3d/bildstein_station1_xyz_intensity_rgb_1K.csv"
#trainingLaserPointsFiles="s3://spark-pointfeatures/Bildstein1/2.txt,s3://spark-pointfeatures/Bildstein1/3.txt"
trainingLaserPointsFiles="./Enschede_1_test_space.csv"
#predictLaserPointsFiles="s3://spark-pointfeatures/Bildstein1/1.txt,s3://spark-pointfeatures/Bildstein1/3.txt" # COMMA separated list
predictLaserPointsFiles="./Enschede_1_test_space.csv"
batchSize=200000
#queryKNNScriptPath="/home/hadoop/queryKNN.py"
queryKNNScriptPath="./queryKNN.py"
#buildKNNObjectScriptPath="/home/hadoop/buildKNNObject.py"
buildKNNObjectScriptPath="./buildKNNObject.py"
buildPickles="1"
kNNpicklePath="."
#savedModelPath="s3://spark-pointfeatures/savedModels/bildstein1.irfmodel"
savedModelPath="./bildstein1.irfmodel"
doEvaluation=0
doTraining=true

# Make sure to avoid "hadoop file/folder exists error"
# Remove existing parquet files
# hdfs dfs -rm -r /user/hadoop/pointFeatures.parquet /user/hadoop/classifedPoints.parquet

# Delete temporary folders created for storing the tiled files before starting the program
for file in $(echo "${predictLaserPointsFiles} ${trainingLaserPointsFiles}" | sed "s/,/ /g" )
do
	#aws s3 rm --recursive ${file}_tiles_folder
	rm -r ./${file}_tiles_folder
	#aws s3 rm --recursive ${file}-classified
	rm -r ./${file}-classified
done

# Train
if [ "$doTraining" == true ]; then
	hdfs dfs -rm -r $savedModelPath
	rm -r $savedModelPath
	$SPARK_HOME/spark-submit --master local[*] --class de.tu_berlin.dima.IRF \
	        ./ml.jar train\
	        $trainingLaserPointsFiles $batchSize $queryKNNScriptPath $buildKNNObjectScriptPath $buildPickles \
	        $kNNpicklePath $savedModelPath $doEvaluation
fi


# Clean up: delete temporary folders created for storing the tiled files
for file in $(echo "${trainingLaserPointsFiles}" | sed "s/,/ /g" )
do
	#aws s3 rm --recursive ${file}_tiles_folder
	rm -r ./${file}_tiles_folder
done

# Predict
$SPARK_HOME/spark-submit --master local[*] --class de.tu_berlin.dima.IRF \
        ./ml.jar predict\
        $predictLaserPointsFiles $batchSize $queryKNNScriptPath $buildKNNObjectScriptPath $buildPickles \
        $kNNpicklePath $savedModelPath $doEvaluation

# Clean up: delete temporary folders created for storing the tiled files
for file in $(echo "${predictLaserPointsFiles}" | sed "s/,/ /g" )
do
	#aws s3 rm --recursive ${file}_tiles_folder
	rm -r ./${file}_tiles_folder
done

