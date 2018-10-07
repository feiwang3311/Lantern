package lantern

import org.scalatest.FunSuite

class LanternFunSuite extends FunSuite {
  def runTest(driver: LanternDriver[String, Unit]) {
    driver.eval("dummy")
  }
}