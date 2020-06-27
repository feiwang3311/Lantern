package lantern.collection.mutable

import lms.core.stub._
import lms.core.virtualize
import lms.macros.SourceContext
import lms.core.utils
import lms.TutorialFunSuite


class StackArrayTest extends TutorialFunSuite {

  val under = "third-party/"

  abstract class DslDriverCStackArray[A:Manifest, B:Manifest] extends DslDriverC[A,B] with StackArrayOps { q =>
      override val codegen = new DslGenC with CCodeGenStackArray {
        val IR: q.type = q
      }
  }

  test("stack_array") {
    val driver = new DslDriverCStackArray[Int,Unit] {
      @virtualize
      def snippet(arg: Rep[Int]) = {
        val x = NewStackArray[Int](10)
        x(0) = arg
        printf("%d", x(1))
      }
    }
    check("stack_array", driver.code, "c")
  }
}