name := "Lantern"

version := "1.0"

scalaVersion := "2.12.4"

resolvers += Resolver.sonatypeRepo("snapshots")

// libraryDependencies += "org.scala-lang.lms" %% "lms-core-macrovirt" % "0.9.0-SNAPSHOT"

libraryDependencies += "org.scalatest" % "scalatest_2.12" % "3.0.4"

libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value % "compile"

autoCompilerPlugins := true

val paradiseVersion = "2.1.0"

addCompilerPlugin("org.scalamacros" % "paradise" % paradiseVersion cross CrossVersion.full)

// tests are not thread safe
parallelExecution in Test := false
// concurrentRestrictions in Global += Tags.limit(Tags.Test, 1)

lazy val lantern = (project in file(".")).dependsOn(lms % "test->test; compile->compile")

lazy val lms = ProjectRef(file("../lms-clean"), "lms-clean")

//cps

addCompilerPlugin("org.scala-lang.plugins" % "scala-continuations-plugin_2.12.0" % "1.0.3")

libraryDependencies += "org.scala-lang.plugins" % "scala-continuations-library_2.12" % "1.0.3"

scalacOptions += "-P:continuations:enable"

// onnx

libraryDependencies ++= Seq("org.bytedeco.javacpp-presets" % "onnx-platform" % "1.3.0-1.4.3")
