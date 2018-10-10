package lantern

import org.scalactic.source
import org.scalatest.{FunSuite, Tag}

class LanternFunSuite extends FunSuite {
  def runTest(driver: LanternDriver[String, Unit]) {
    driver.eval("dummy")
  }

  // TODO: Edit this function to actually detect whether GPU codegen is possible.
  // One idea: check for:
  // - The existence of cuBLAS header files (<cuda_runtime.h>, <cublas_v2.h>).
  // - The existence of a GPU (perhaps run `nvidia-smi`).
  def isGPUAvailable = false

  // Utility function wrapping `test` that checks whether GPU is available.
  def testGPU(testName: String, testTags: Tag*)(testFun: => Any /* Assertion */)(implicit pos: source.Position) {
    if (isGPUAvailable)
      test(testName, testTags: _*)(testFun)(pos)
    else
      ignore(testName, testTags: _*)(testFun)(pos)
  }
}