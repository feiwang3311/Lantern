package lantern

import sys.process._
import org.scalactic.source
import org.scalatest.{FunSuite, Tag}

class LanternFunSuite extends FunSuite {
  def runTest(driver: LanternDriver[String, Unit]) {
    driver.eval("dummy")
  }

  private val _currentTestName = new ThreadLocal[String]

  override def withFixture(test: NoArgTest) = {
    _currentTestName.set(test.name)
    val outcome = super.withFixture(test)
    _currentTestName.set(null)
    outcome
  }
  protected def currentTestName: String = _currentTestName.get()

  // Returns true if GPU code generation is possible.
  // Currently, checks if `nvcc` exists.
  // One can force GPU code generation by defining the "LANTERN_RUN_GPU" environment variable.
  def isGPUAvailable: Boolean = {
    try {
      ("nvcc --version": ProcessBuilder).!!; true
    } catch {
      case _: Throwable => sys.env.get("LANTERN_RUN_GPU").isDefined
    }
  }

  // Utility function wrapping `test` that checks whether GPU is available.
  def testGPU(testName: String, testTags: Tag*)(testFun: => Any /* Assertion */)(implicit pos: source.Position) {
    if (isGPUAvailable)
      test(testName, testTags: _*)(testFun)(pos)
    else
      ignore(testName, testTags: _*)(testFun)(pos)
  }
}
