name := "Lantern"

version := "1.0"

scalaVersion := "2.12.7"

// cps

autoCompilerPlugins := true

addCompilerPlugin("org.scala-lang.plugins" % "scala-continuations-plugin_2.12.0" % "1.0.3")

libraryDependencies += "org.scala-lang.plugins" % "scala-continuations-library_2.12" % "1.0.3"

scalacOptions += "-P:continuations:enable"

// lms

//scalaOrganization := "org.scala-lang.virtualized"

//scalacOptions += "-Yvirtualize"

resolvers += Resolver.sonatypeRepo("snapshots")

libraryDependencies += "org.scala-lang.lms" %% "lms-core-macrovirt" % "0.9.0-SNAPSHOT"

libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value % "compile"

//libraryDependencies += "org.scalactic" %% "scalactic" % scalaVersion.value

libraryDependencies ++= Seq("org.bytedeco.javacpp-presets" % "onnx-platform" % "1.3.0-1.4.3")

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test"

val paradiseVersion = "2.1.0"

addCompilerPlugin("org.scalamacros" % "paradise" % paradiseVersion cross CrossVersion.full)

//assemblyJarName in assembly := "lantern.jar"
//test in assembly := {}

// libraryDependencies += "org.scala-lang.lms" %% "lms-core" % "1.0.0-SNAPSHOT"

// libraryDependencies += "org.scala-lang.virtualized" % "scala-compiler" % "2.11.2"

// libraryDependencies += "org.scala-lang.virtualized" % "scala-library" % "2.11.2"

// libraryDependencies += "org.scala-lang.virtualized" % "scala-reflect" % "2.11.2"

// set the main class for packaging the main jar
// mainClass in (Compile, packageBin) := Some("scala.lantern")

// scala support for protobuf (for adding onnx support)
//PB.targets in Compile := Seq(
//  scalapb.gen() -> (sourceManaged in Compile).value
//)

// (optional) If you need scalapb/scalapb.proto or anything from
// google/protobuf/*.proto
//libraryDependencies += "com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf"

// Testing.

// Parallel execution is disabled because it causes non-deterministic failures:
// https://github.com/feiwang3311/Lantern/issues/19

// parallelExecution in Test := false
concurrentRestrictions in Global += Tags.limit(Tags.Test, 1)
