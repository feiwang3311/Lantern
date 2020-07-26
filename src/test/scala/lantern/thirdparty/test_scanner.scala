package lms
package thirdparty

import lms.core.stub._
import lms.core.virtualize
import macros.SourceContext
import lms.collection._

class ScannerTest extends TutorialFunSuite {
  val under = "thirdparty/scanner"

  abstract class DslDriverCScanner[A: Manifest, B: Manifest] extends DslDriverC[A,B] with ScannerOps { q =>
    override val codegen = new DslGenC with CCodeGenScannerOps {
      val IR: q.type = q
    }
  }

  test("open") {
    val driver = new DslDriverCScanner[Int, Unit] {
      @virtualize
      def snippet(arg: Rep[Int]) = {
        val lmsPath = System.getProperty("user.dir")
        val filename = lmsPath + "/src/test/scala/lantern/thirdparty/test_scanner.scala"
        val fd = open(filename)
        val filelength = filelen(fd)
        printf("file length is %ld\n", filelength)
        close(fd)
      }
    }
    driver.eval(0)
  }

  test("mmap") {
    val driver = new DslDriverCScanner[Int, Unit] {
      @virtualize
      def snippet(arg: Rep[Int]) = {
        val lmsPath = System.getProperty("user.dir")
        val filename = lmsPath + "/src/test/scala/lantern/thirdparty/test_scanner.scala"
        val fd = open(filename)
        val array = mmap[Char](fd, unit(50l))
        for (i <- (0 until 10): Rep[Range])
          printf("%c", array(i))
        close(fd)
      }
    }
    driver.eval(0)
  }

  test("fopen") {
    val driver = new DslDriverCScanner[Int, Unit] {
      @virtualize
      def snippet(arg: Rep[Int]) = {
        val lmsPath = System.getProperty("user.dir")
        val filename = lmsPath + "/src/test/scala/lantern/thirdparty/test_binary"
        val fp = fopen(filename, "w")
        fprintf(fp, "%d", 10)
        fclose(fp)

        val fp2 = fopen(filename, "r")
        var target = 0
        getInt(fp2, target)
        printf("%d", target)
        fclose(fp2)
      }
    }
    driver.eval(0)
  }
}

