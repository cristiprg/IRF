name := "ML Project"

version := "1.0"

scalaVersion := "2.10.4"

test in assembly := {}
assemblyJarName in assembly := "ml.jar"

// unmanagedBase := baseDirectory.value / "lib"

// libraryDependencies += "org.apache.commons" % "commons-math3" % "3.4.1"
// libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "1.5.1"
// libraryDependencies += "org.apache.spark" % "spark-graphx_2.11" % "1.5.1"
// libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "1.5.1"
// libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % "1.5.1"
// libraryDependencies += "org.jpmml" % "pmml-model" % "1.1.15"
// libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.11.2"


// libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.3" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.6.3" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.3" % "provided"

//libraryDependencies += "org.eclipse.jetty" % "jetty-webapp" % "9.1.0.v20131115"
//libraryDependencies += "org.eclipse.jetty" % "jetty-plus" % "9.1.0.v20131115"
//libraryDependencies += "org.eclipse.jetty" % "jetty-client" % "9.2.10.v20150310"
//
//libraryDependencies += "org.mongodb" %% "casbah" % "3.0.0"
libraryDependencies += "com.databricks" %% "spark-csv" % "1.5.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"
libraryDependencies +=  "org.scalaj" %% "scalaj-http" % "2.0.0"

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

////////////////////////
// http://queirozf.com/entries/creating-scala-fat-jars-for-spark-on-sbt-with-sbt-assembly-plugin

//libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.0" % "provided"
//libraryDependencies += "org.apache.spark" %% "spark-streaming" % "1.6.0" % "provided"
//libraryDependencies += "org.apache.spark" %% "spark-streaming-kinesis-asl" % "1.6.0"
//libraryDependencies += "com.amazonaws" % "amazon-kinesis-client" % "1.6.1"
//libraryDependencies += "com.amazonaws" % "amazon-kinesis-producer" % "0.10.2"

assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs @ _*) => MergeStrategy.last
  case PathList("javax", "activation", xs @ _*) => MergeStrategy.last
  case PathList("org", "apache", xs @ _*) => MergeStrategy.last
  case PathList("com", "google", xs @ _*) => MergeStrategy.last
  case PathList("com", "esotericsoftware", xs @ _*) => MergeStrategy.last
  case PathList("com", "codahale", xs @ _*) => MergeStrategy.last
  case PathList("com", "yammer", xs @ _*) => MergeStrategy.last
  case "about.html" => MergeStrategy.rename
  case "META-INF/ECLIPSEF.RSA" => MergeStrategy.last
  case "META-INF/mailcap" => MergeStrategy.last
  case "META-INF/mimetypes.default" => MergeStrategy.last
  case "plugin.properties" => MergeStrategy.last
  case "log4j.properties" => MergeStrategy.last
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}
