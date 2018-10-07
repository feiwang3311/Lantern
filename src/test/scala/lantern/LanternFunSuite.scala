package lantern

import org.scalatest.FunSuite

class LanternFunSuite extends FunSuite {
  def runTest(snippet: DslDriverBase[String, Unit]) {
    snippet.eval("dummy")
  }
}