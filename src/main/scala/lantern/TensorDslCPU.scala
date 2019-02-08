package lantern

import scala.util.continuations._
import org.scala_lang.virtualized.virtualize
import org.scala_lang.virtualized.SourceContext

import scala.virtualization.lms._
import scala.virtualization.lms.common._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.{Map => MutableMap}
import scala.math._

trait TensorDslCPU extends TensorDsl {

  class BackendCPU protected() extends Backend {
    override def setup() {}
    override def cleanup() {}
    override def mallocArray[T: Manifest](length: Rep[Int]): Rep[Array[T]] = NewArray[T](length)

    override def copyFloatArray(dest: Rep[Array[Float]], src: Rep[Array[Float]], length: Rep[Int]): Unit = {
      for (i <- DataLoop(length)) dest(i) = src(i)
    }

    override def arrayToTensor(array: Rep[Array[Float]], dims: Rep[Int]*): Tensor = new Tensor(array, dims)

    override def makeTensor(dims: Seq[Rep[Int]], scalars: Float*): Tensor = {
      Tensor(Array(scalars.map(unit(_)): _*), dims: _*)
    }

    override def fill(dims: Seq[Rep[Int]], value: Rep[Float]): Tensor = {
      val scalarCount = dims.product1
      val array = mallocArray[Float](scalarCount)
      for (i <- DataLoop(scalarCount)) array(i) = value
      Tensor(array, dims: _*)
    }

    // TODO (Optimizations) (Fei Wang): bias has the feature that the values before bias is never used otherwise
    // The consequence is that add bias can be done in-place with broadcasting
    // and backprop to bias can be done by += with reduction
    // In that sense, this function should be removed, and we should use plusBias/plusBias_grad instead
    override def fillWithBias(dims: Seq[Rep[Int]], bias: Tensor, dim: Int): Tensor = {
      assert(dim >= 0 && dim < dims.size, s"Target dimension $dim is out of range $dims")
      assert(bias.rank == 1 && bias.scalarCount == dims(dim),
        "Bias must be 1D and have length equal to the target dimension")
      val scalarCount: Rep[Int] = dims.product1
      val res = mallocArray[Float](scalarCount)

      // iterate for higherDims
      val offset = var_new(0)
      for (hd <- DataLoop(dims.take(dim).product1)) {
        // iterate for current dim
        for (cd <- DataLoop(dims.drop(dim).head)) {
          // iterate for lowerDims
          for (ld <- DataLoop(dims.drop(dim+1).product1)) {
            res(offset) = bias.data(cd)
            offset += 1
          }
        }
      }
      Tensor(res, dims: _*)
    }

    // TODO (Optimization) (Fei Wang): It is advisable that all mapping like functions (fillInPlace, map, mapInplace)
    // should take a function/closure that starts from index (i => compute_value_at_pos_i)
    override def fillInPlace(x: Tensor, value: Rep[Float]): Unit = {
      for (i <- DataLoop(x.scalarCount)) x.data(i) = value
    }

    def fillByLinearIndex(x: Tensor, func: (Rep[Int] => Rep[Float])): Unit = {
      for (i <- DataLoop(x.scalarCount)) x.data(i) = func(i)
    }

    // TODO (Need Dependent Type for func??)
    @virtualize
    def fillByStepIndex(x: Tensor, func: (Seq[Rep[Int]] => Rep[Float])): Unit = {
      def write(shape: Seq[Rep[Int]], index: Seq[Rep[Int]]): Unit = {
        for (i <- (0 until shape(0)))
          if (shape.size == 1) {
            val idx = (x.shape.strides zip index).foldLeft(i){case (a, (b, c)) => a + b * c}
            x.data(idx) = func(index :+ i)
          } else {
            write(shape.tail, index :+ i)
          }
      }
      write(x.shape, Seq[Rep[Int]]())
    }

    @virtualize
    def traverseShapeByStepIndex(x: Dimensions, func: (Seq[Rep[Int]] => Unit)): Unit = {
      def act(shape: Seq[Rep[Int]], index: Seq[Rep[Int]]): Unit = {
        for (i <- (0 until shape(0)))
          if (shape.size == 1) {
            func(index :+ i)
          } else {
            act(shape.tail, index :+ i)
          }
      }
      act(x.dims, Seq[Rep[Int]]())
    }

    override def randinit(dims: Seq[Int], scale: Float = 1.0f, seed: Option[Int] = None): Tensor = {
      seed match {
        case None => ()
        case Some(seed) => Random.srand(Some(seed))
      }
      val scalarCount = dims.product
      val res = mallocArray[Float](scalarCount)
      for (i <- DataLoop(scalarCount)) res(i) = (Random.rand() - 0.5f) * scale
      new Tensor(res, dims)
    }

    @virtualize
    override def clipAt(x: Tensor, bound: Float) = {
      for (i <- DataLoop(x.scalarCount)) {
        val temp = x.data(i)
        if (temp > bound) x.data(i) = bound
        if (temp < -1.0f * bound) x.data(i) = -1.0f * bound
      }
    }

    override def mutate(x: Tensor, delta: Rep[Int] => Rep[Float]): Unit = for (i <- DataLoop(x.scalarCount)) x.data(i) += delta(i)
    override def mapInPlace(x: Tensor, op: Rep[Float] => Rep[Float]): Unit = for (i <- DataLoop(x.scalarCount)) x.data(i) = op(x.data(i))
    override def changeTo(x: Tensor, gen: Rep[Int] => Rep[Float]): Unit = for (i <- DataLoop(x.scalarCount)) x.data(i) = gen(i)
    override def map(x: Tensor, op: Rep[Float] => Rep[Float]): Tensor = {
      val res = mallocArray[Float](x.scalarCount)
      for (i <- DataLoop(x.scalarCount)) res(i) = op(x.data(i))
      new Tensor(res, x.shape)
    }
    override def fold(init: Rep[Float])(x: Tensor, op: (Rep[Float], Rep[Float]) => Rep[Float]): Rep[Float] = {
      val res = var_new[Float](init)
      for (i <- DataLoop(x.scalarCount)) var_assign(res, op(res, x.data(i)))
      res
    }

    override def vectorVectorDot(x: Tensor, y: Tensor): Tensor = {
      assertC(x.shape(0) == y.shape(0), "vector vector dot not the same %d %d", x.shape(0), y.shape(0))
      val value = var_new(0.0f)
      for (i <- DataLoop(x.shape.last)) {
        value += x.data(i) * y.data(i)
      }
      val res = mallocArray[Float](1)
      res(0) = readVar(value)
      Tensor(res, 1)
    }

    override def matrixVectorDot(x: Tensor, y: Tensor): Tensor = {
      assertC(x.shape(1) == y.shape(0), "matrix vector dot dim1 of x (%d) is not the same with dim0 of y (%d)", x.shape(1), y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val res = mallocArray[Float](dim1)
      unchecked[Unit] (
        "cblas_sgemv(CblasRowMajor, CblasNoTrans, ",
        dim1, ",", dim2, ",", 1, ",",
        x.data, ",", dim2, ",", y.data, ",", 1, ",", 0, ",", res, ",", 1, ")")
      Tensor(res, dim1)
    }

    override def matrixMatrixDot(x: Tensor, y: Tensor): Tensor = {
      assertC(x.shape(1) == y.shape(0), "matrix matrix dot dim1 of x (%d) is not the same with dim0 of y (%d)", x.shape(1), y.shape(0))
      val dim1 = x.shape(0)
      val dim2 = x.shape(1)
      val dim3 = y.shape(1)
      val res = mallocArray[Float](dim1 * dim3)
      unchecked[Unit](
        "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
        dim1, ",", dim3, ",", dim2, ",", 1, ",",
        x.data, ",", dim2, ",", y.data, ",", dim3, ",", 0, ",", res, ",", dim3, ")")
      Tensor(res, dim1, dim3)
    }

    override def dot_grad(x: TensorR, y: TensorR, output: TensorR): Unit = {
      (x.x.rank, y.x.rank) match {
        case (1, 1) =>
          if (!x.isInput) x.d.addMul(output.d.data(0), y.x)
          if (!y.isInput) y.d.addMul(output.d.data(0), x.x)
        case (2, 1) =>
          if (!x.isInput) add_cartesian(x.d, y.x, output.d); // that.d.add_composion(this.x, y.d)
          if (!y.isInput) {
            val dim1 = x.x.shape(0); val dim2 = x.x.shape(1)
            unchecked[Unit](
              "cblas_sgemv(CblasRowMajor, CblasTrans, ",
              dim1, ",", dim2, ",", 1, ",",
              x.x.data, ",", dim2, ",", output.d.data, ",", 1, ",", 1, ",", y.d.data, ",", 1, ")")
          }
        case (2, 2) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          generateRawComment("backprop of matrix-matrix-dot")
          if (!x.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim1, ",", dim2, ",", dim3, ",", 1, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim2, ")")
          if (!y.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim2, ",", dim3, ",", dim1, ",", 1, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", 1, ",", y.d.data, ",", dim3, ")")
      }
    }

    // setting: this is matrix, that is dims(0)-sized vector, y is dims(1)-sized vector
    // the result is to update this so that this += that * y, where * is Cartesian product
    override def add_cartesian(x: Tensor, y: Tensor, output: Tensor) = {
      generateRawComment("backend add_cartesian")
      assert(x.rank == 2 && y.shape == Dimensions(Seq(x.shape(1))) && output.shape == Dimensions(Seq(x.shape(0))))
      val off = var_new(0)
      for (i <- DataLoop(x.shape(0))) {
        for (j <- DataLoop(x.shape(1))) {
          x.data(off + j) = x.data(off + j) + y.data(j) * output.data(i)
        }
        off += x.shape(1)
      }
    }

    @virtualize
    def elementWiseOpWithBroadCast(x: Tensor, y: Tensor, op: ((Rep[Float], Rep[Float]) => Rep[Float])) = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case Some((xShape, yShape, resShape)) => {
          val resData = mallocArray[Float](resShape.scalarCount)
          val res = new Tensor(resData, resShape)
          val xStridesShadow = (xShape.strides zip xShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          val yStridesShadow = (yShape.strides zip yShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          fillByStepIndex(res, {idx: Seq[Rep[Int]] =>
            val idxX = (xStridesShadow zip idx).foldLeft(unit(0)){case (a, (b, c)) => a + b * c}
            val idxY = (yStridesShadow zip idx).foldLeft(unit(0)){case (a, (b, c)) => a + b * c}
            op(x.data(idxX), y.data(idxY))
          })
          (res, xShape, yShape)
        }
        case _ => ???
      }
    }

    type RF3 = ((Rep[Float], Rep[Float], Rep[Float]) => Rep[Float])
    @virtualize // (fuse gradient updates of both operands
    def backpropElementWiseOpWithBroadCast(in1: TensorR, in2: TensorR, out: TensorR, op1: RF3, op2: RF3): Unit = {
      Tensor.dimBroadcast(in1.x.shape, in2.x.shape) match {
        case Some((xShape, yShape, resShape)) => {
          val xStridesShadow = (xShape.strides zip xShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          val yStridesShadow = (yShape.strides zip yShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          traverseShapeByStepIndex(resShape, {idx: Seq[Rep[Int]] =>
            val idxX = (xStridesShadow zip idx).foldLeft(unit(0)){case (a, (b, c)) => a + b * c}
            val idxY = (yStridesShadow zip idx).foldLeft(unit(0)){case (a, (b, c)) => a + b * c}
            val idxR = (resShape.strides zip idx).foldLeft(unit(0)){case (a, (b, c)) => a + b * c}
            if (!in1.isInput) in1.d.data(idxX) += op1(in1.x.data(idxX), in2.x.data(idxY), out.d.data(idxR))
            if (!in2.isInput) in2.d.data(idxY) += op2(in1.x.data(idxX), in2.x.data(idxY), out.d.data(idxR))
          })
        }
        case _ => ???
      }
    }

    @virtualize
    // x += op(x, y) (with potentially broadcasting y, or reducing y (reverse of broadcasting x))
    def inplaceElementWiseOpWithBroadCastOrReduce(x: Tensor, y: Tensor, op: ((Rep[Float], Rep[Float]) => Rep[Float])): Unit = {
      Tensor.dimBroadcast(x.shape, y.shape) match {
        case Some((xShape, yShape, resShape)) => {
          val xStridesShadow = (xShape.strides zip xShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          val yStridesShadow = (yShape.strides zip yShape.dims) map {case (a, b) => if (b == unit(1)) 0 else a}
          traverseShapeByStepIndex(resShape, {idx: Seq[Rep[Int]] =>
            val idxX = (xStridesShadow zip idx).foldLeft(unit(0)){case (a, (b, c)) => a + b * c}
            val idxY = (yStridesShadow zip idx).foldLeft(unit(0)){case (a, (b, c)) => a + b * c}
            x.data(idxX) = op(x.data(idxX), y.data(idxY))
          })
        }
      }
    }

    override def plusBias(main: Tensor, bias: Tensor): Tensor = {
      this.inplaceElementWiseOpWithBroadCastOrReduce(main, bias, (_ + _))
      main
    }

    override def plusBias_grad(main: TensorR, bias: TensorR): Unit = {
      if (!bias.isInput) this.inplaceElementWiseOpWithBroadCastOrReduce(bias.d, main.d, (_ + _))
    }

    override def plusEqual(base: Tensor, adder: Tensor): Tensor = {
      this.inplaceElementWiseOpWithBroadCastOrReduce(base, adder, (_ + _))
      base
    }
    override def plusEqual_grad(base: TensorR, adder: TensorR): Unit = {
      if (!adder.isInput) this.inplaceElementWiseOpWithBroadCastOrReduce(adder.d, base.d, (_ + _))
    }

    override def +(x: Tensor, y: Rep[Float]): Tensor = map(x, s => s + y)
    override def +(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseOpWithBroadCast(x, y, _ + _)
    override def add_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val op1 = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      val op2 = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      backpropElementWiseOpWithBroadCast(x, y, output, op1, op2)
    }

    override def +=(x: Tensor, y: Rep[Float]): Unit = mapInPlace(x, s => s + y)
    override def +=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ + _))

    override def -(x: Tensor, y: Rep[Float]): Tensor = map(x, s => s - y)
    override def -(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseOpWithBroadCast(x, y, _ - _)
    override def minus_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val op1 = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => c
      val op2 = (_: Rep[Float], _: Rep[Float], c: Rep[Float]) => 0.0f - c
      backpropElementWiseOpWithBroadCast(x, y, output, op1, op2)
    }

    override def -=(x: Tensor, y: Rep[Float]): Unit = mapInPlace(x, s => s - y)
    override def -=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ - _))

    override def *(x: Tensor, y: Rep[Float]): Tensor = map(x, s => s * y)
    override def *(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseOpWithBroadCast(x, y, _ * _)
    override def mul_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val op1 = (_: Rep[Float], b: Rep[Float], c: Rep[Float]) => c * b
      val op2 = (a: Rep[Float], _: Rep[Float], c: Rep[Float]) => c * a
      backpropElementWiseOpWithBroadCast(x, y, output, op1, op2)
    }

    override def *=(x: Tensor, y: Rep[Float]): Unit = mapInPlace(x, s => s * y)
    override def *=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ * _))

    override def /(x: Tensor, y: Rep[Float]): Tensor = map(x, s => s / y)
    override def /(x: Tensor, y: Tensor): (Tensor, Dimensions, Dimensions) = elementWiseOpWithBroadCast(x, y, _ / _)
    override def div_grad(x: TensorR, y: TensorR, output: TensorR, xShape: Dimensions, yShape: Dimensions): Unit = {
      val op1 = (_: Rep[Float], b: Rep[Float], c: Rep[Float]) => c / b
      val op2 = (a: Rep[Float], b: Rep[Float], c: Rep[Float]) => -1.0f * a * c / (b * b)
      backpropElementWiseOpWithBroadCast(x, y, output, op1, op2)
    }

    override def /=(x: Tensor, y: Rep[Float]): Unit = mapInPlace(x, s => s / y)
    override def /=(x: Tensor, y: Tensor): Unit = inplaceElementWiseOpWithBroadCastOrReduce(x, y, (_ / _))

    override def geam(x: Tensor, transX: Boolean, alpha: Rep[Float], y: Tensor, transY: Boolean, beta: Rep[Float], output: Tensor): Unit = {
      (transX, transY) match {
        case (false, false) => output.changeTo { i => x.data(i) * alpha + y.data(i) * beta }
        case _ => ???
      }
    }

    override def trans(x: Tensor): Tensor = {
      assert(x.rank == 2, "transpose is only for matrix. Tensor transpose is not supported here")
      val res = backend.mallocArray[Float](x.scalarCount)
      val offT = var_new(0)
      for (i <- DataLoop(x.shape(1))) {
        val off = var_new(0)
        for (j <- DataLoop(x.shape(0))) {
          res(offT + j) = x.data(off + i)
          off += x.shape(1)
        }
        offT += x.shape(0)
      }
      new Tensor(res, x.shape.reverse)
    }

    override def trans_grad(x: TensorR, y: TensorR): Unit = {
      val offT = var_new(0)
      for (i <- DataLoop(x.x.shape(1))) {
        val off = var_new(0)
        for (j <- DataLoop(x.x.shape(0))) {
          x.d.data(off + i) += y.d.data(offT + j)
          off += x.x.shape(1)
        }
        offT += x.x.shape(0)
      }
    }

    override def permute(x: Tensor, dims: Int*): Tensor = ???
    override def permute_grad(x: TensorR, y: TensorR, dims: Int*): Unit = ???

    override def gemm(x: Tensor, transX: Boolean, y: Tensor, transY: Boolean, alpha: Float): Tensor = {
      (transX, transY) match {
        case (false, false) =>
          assert(x.shape(1) == y.shape(0))
          val dim1 = x.shape(0)
          val dim2 = x.shape(1)
          val dim3 = y.shape(1)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim2, ",", y.data, ",", dim3, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
        case (false, true) =>
          assert(x.shape(1) == y.shape(1))
          val dim1 = x.shape(0)
          val dim2 = x.shape(1)
          val dim3 = y.shape(0)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim2, ",", y.data, ",", dim2, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
        case (true, false) =>
          assert(x.shape(0) == y.shape(0), s"gemm dims don't match, got ${x.shape.seq}, ${y.shape.seq}")
          val dim1 = x.shape(1)
          val dim2 = x.shape(0)
          val dim3 = y.shape(1)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim1, ",", y.data, ",", dim3, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
        case (true, true) =>
          assert(x.shape(0) == y.shape(1))
          val dim1 = x.shape(1)
          val dim2 = x.shape(0)
          val dim3 = y.shape(0)
          val res = mallocArray[Float](dim1 * dim3)
          unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ",
            dim1, ",", dim3, ",", dim2, ",", alpha, ",",
            x.data, ",", dim1, ",", y.data, ",", dim2, ",", 0, ",", res, ",", dim3, ")")
          Tensor(res, dim1, dim3)
      }
    }

    override def gemm_grad(x: TensorR, transX: Boolean, y: TensorR, transY: Boolean, alpha: Float, output: TensorR): Unit = {
      generateRawComment(s"backprop of gemm ${x.x.shape.seq}, ${transX}, ${y.x.shape.seq}, ${transY}")
      (transX, transY) match {
        case (false, false) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(1)
          if (!x.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim1, ",", dim2, ",", dim3, ",", alpha, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim2, ")")
          if (!y.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim2, ",", dim3, ",", dim1, ",", alpha, ",",
            x.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", 1, ",", y.d.data, ",", dim3, ")")
        case (false, true) =>
          val dim1 = x.x.shape(0); val dim2 = x.x.shape(1); val dim3 = y.x.shape(0)
          if (!x.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
            dim1, ",", dim2, ",", dim3, ",", alpha, ",",
            output.d.data, ",", dim3, ",", y.x.data, ",", dim2, ",", 1, ",", x.d.data, ",", dim2, ")")
          if (!y.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
            dim3, ",", dim2, ",", dim1, ",", alpha, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim2, ",", 1, ",", y.d.data, ",", dim2, ")")
        case (true, false) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(1)
          if (!x.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
            dim2, ",", dim1, ",", dim3, ",", alpha, ",",
            y.x.data, ",", dim3, ",", output.d.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim1, ")")
          if (!y.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
            dim2, ",", dim3, ",", dim1, ",", alpha, ",",
            x.x.data, ",", dim1, ",", output.d.data, ",", dim3, ",", 1, ",", y.d.data, ",", dim3, ")")
        case (true, true) =>
          val dim1 = x.x.shape(1); val dim2 = x.x.shape(0); val dim3 = y.x.shape(0)
          if (!x.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ",
            dim2, ",", dim1, ",", dim3, ",", alpha, ",",
            y.x.data, ",", dim2, ",", output.d.data, ",", dim3, ",", 1, ",", x.d.data, ",", dim1, ")")
          if (!y.isInput) unchecked[Unit](
            "cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ",
            dim3, ",", dim2, ",", dim1, ",", alpha, ",",
            output.d.data, ",", dim3, ",", x.x.data, ",", dim1, ",", 1, ",", y.d.data, ",", dim2, ")")
      }
    }

    // implementation of Conv2D following Pytorch's idea (transform conv2d into matrix-matrix-dot, and use OpenBLAS)
    // https://github.com/pytorch/pytorch/blob/0a8c8c1dbead2f845e524ae32c19167d80363148/aten/src/THNN/generic/SpatialConvolutionMM.c
    type RAF = Rep[Array[Float]]
    def memsetFloatZero(where: RAF, howmany: Rep[Int]) = {
      unchecked[Unit]("memset(", where, ", 0, 4 * ", howmany, ");")
    }
    def memcpyFloat(dst: RAF, src: RAF, howmany: Rep[Int]) = {
      unchecked[Unit]("memcpy(", dst, ", ", src, ", 4 * ", howmany, ");")
    }

    def unfoldedCopy(finput: RAF, input: RAF, kW: Rep[Int], kH: Rep[Int], dW: Int, dH: Int, padW: Int, padH: Int,
    nInputPlane: Rep[Int], inputWidth: Rep[Int], inputHeight: Rep[Int], outputWidth: Rep[Int], outputHeight: Rep[Int]) {
      for (k <- (0 until nInputPlane * kH * kW): Rep[Range]) {
        val nip = k / (kH * kW)
        val rest = k % (kH * kW)
        val kh = rest / kW
        val kw = rest % kW
        val dst = slice(finput, nip*kH*kW*outputHeight*outputWidth + kh*kW*outputHeight*outputWidth + kw*outputWidth*outputWidth)
        val src = slice(input,  nip*inputHeight*inputWidth)
        if (padW > 0 || padH > 0) {
          for (y <- (0 until outputHeight): Rep[Range]) {
            val iy = y * dH - padH + kh
            __ifThenElse ((iy < 0 || iy >= inputHeight), {
              memsetFloatZero(slice(dst, y*outputWidth), outputWidth); ()
            }, {
              if (dW == 1) {
                val ix = 0 - padW + kw;
                val lpad = __ifThenElse ((padW-kw > 0), padW-kw, 0)
                val rpad = __ifThenElse ((padW-(kW-kw-1) > 0), padW-(kW-kw-1), 0)
                __ifThenElse ((outputWidth-rpad-lpad <= 0), {
                  memsetFloatZero(slice(dst, y*outputWidth), outputWidth)
                }, {
                  __ifThenElse ((lpad > 0), memsetFloatZero(slice(dst, y*outputWidth), lpad), ())
                  generateRawComment("may have segfault here")
                  memcpyFloat(slice(dst, y*outputWidth+lpad), slice(src, iy*inputWidth+ix+lpad), outputWidth-rpad-lpad)
                  __ifThenElse ((rpad > 0), memsetFloatZero(slice(dst, y*outputWidth+outputWidth-rpad), rpad), ())
                })
              } else {
                for (x <- (0 until outputWidth): Rep[Range]) {
                  val ix = x * dW - padW + kw
                  __ifThenElse ((ix < 0 || ix >= inputWidth), memsetFloatZero(slice(dst, y*outputWidth+x), 1),
                    memcpyFloat(slice(dst, y*outputWidth+x), slice(src, iy*inputWidth+ix), 1))
                }
              }
            })
          }
        } else {
          for (y <- (0 until outputHeight): Rep[Range]) {
            val iy = y * dH + kh
            val ix = kw
            if (dW == 1) memcpyFloat(slice(dst, y*outputWidth), slice(src, iy*inputWidth+ix), outputWidth)
            else for (x <- (0 until outputWidth): Rep[Range])
              memcpyFloat(slice(dst, y*outputWidth+x), slice(src, iy*inputWidth+ix+x*dW), 1)
          }
        }
      }
    }

    override def conv2D_batch(input: Tensor, kernel: Tensor, bias: Option[Tensor], strides: Seq[Int], pads: Seq[Int]): (Tensor, Option[Tensor]) = {
      val ((dH:Int) :: (dW:Int) :: Nil) = strides.take(2).toList
      val (padH, padW) = if (pads.size == 1) (pads(0), pads(0)) else {if (pads.size == 2) (pads(0), pads(1)) else if (pads.size == 4) (pads(0), pads(2)) else ???}
      val nOutputPlane = kernel.shape(0)
      val kH = kernel.shape(2)
      val kW = kernel.shape(3)
      val batchSize = input.shape(0)
      val nInputPlane = input.shape(1)
      val inputHeight = input.shape(2)
      val inputWidth = input.shape(3)
      val outputHeight = (inputHeight + 2*padH - kH) / dH + 1
      val outputWidth  = (inputWidth + 2*padW - kW) / dW + 1
      val output = bias match {
          case Some(bias) => Tensor.fillWithBias(Seq(input.shape(0), kernel.shape(0), outputHeight, outputWidth), bias, 1)
          case None => Tensor.zeros(input.shape(0), kernel.shape(0), outputHeight, outputWidth)
        }
      val finput = Tensor.zeros(batchSize, kW * kH * nInputPlane, outputHeight * outputWidth)
      for (t <- (0 until batchSize): Rep[Range]) {
        val input_t = input(t).data
        val output_t = output(t).data
        val finput_t = finput(t).data
        ConvOutputFrame(input_t, output_t, kernel.data, finput_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight)
      }
      (output, Some(finput))
    }

    def ConvOutputFrame(input: RAF, output: RAF, weight: RAF, finput: RAF, kW: Rep[Int], kH: Rep[Int], dW: Int, dH: Int, padW: Int, padH: Int,
      nInputPlane: Rep[Int], inputWidth: Rep[Int], inputHeight: Rep[Int], nOutputPlane: Rep[Int], outputWidth: Rep[Int], outputHeight: Rep[Int]) {

      unfoldedCopy(finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
      // finput viewed as: kW*kH*nInputPlane, outputHeight * outputWidth
      // input  viewed as: nInputPlane, inputWidth, inputHeight
      val dim1 = nOutputPlane
      val dim2 = kW * kH *nInputPlane
      val dim3 = outputHeight * outputWidth
      unchecked[Unit](
        "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ",
        dim1, ",", dim3, ",", dim2, ",", 1, ",",
        weight, ",", dim2, ",", finput, ",", dim3, ",", 1, ",", output, ",", dim3, ")")
    }

    // Gradient of `conv2D_batch`.
    @virtualize
    override def conv2D_batch_grad(input: TensorR, finput: Option[TensorR], filter: TensorR, res: TensorR, bias: Option[TensorR] = None,
                                   padding: (Int, Int), strides: (Int, Int), dilations: (Int, Int)): Unit = {
      // NOTE: Strides/paddings may be in the wrong order.
      assert(dilations._1 == 1 && dilations._2 == 1, "Currently, only dilations of 1 are supported")
      val finputR: TensorR = finput match {
        case None => assert(false, "BackendCPU needs finput to be Some[TensorR], found None"); TensorR(Tensor.zeros(1))
        case Some(finputr) => finputr
      }

      // back-propagate to inputs
      if (!input.isInput) ConvGradInput(res.d, input.d, finputR.d, filter.x, strides._1, strides._2, padding._1, padding._2)
      // back-propagate to weights
      bias match {
        case None => ConvGradParam(finputR.x, res.d, filter.d, None, strides._1, strides._2, padding._1, padding._2)
        case Some(bias) => ConvGradParam(finputR.x, res.d, filter.d, Some(bias.d), strides._1, strides._2, padding._1, padding._2)
      }
    }

    def ConvGradParam(finput: Tensor, gradOutput: Tensor, gradWeight: Tensor, gradBias: Option[Tensor], dH: Int, dW: Int, padH: Int, padW: Int, scale: Float = 1.0f) = {
      val nInputPlane = gradWeight.shape(1)
      val kH = gradWeight.shape(2)
      val kW = gradWeight.shape(3)
      val batchSize = gradOutput.shape(0)
      val nOutputPlane = gradOutput.shape(1)
      val outputHeight = gradOutput.shape(2)
      val outputWidth = gradOutput.shape(3)
      for (t <- (0 until batchSize): Rep[Range]) {
        val gradOutput_t = gradOutput(t).data
        val finput_t = finput(t).data
        val dim1 = nOutputPlane
        val dim2 = outputWidth * outputHeight
        val dim3 = kW * kH * nInputPlane
        unchecked[Unit](
          "cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ",
          dim1, ",", dim3, ",", dim2, ",", scale, ",",
          gradOutput_t, ",", dim2, ",", finput_t, ",", dim2, ",", 1, ",", gradWeight.data, ",", dim3, ")")
        gradBias match {
          case None => ()
          case Some(gradBias) =>
            for (i <- (0 until nOutputPlane): Rep[Range]) {
              val sum = var_new(0.0f)
              val data = slice(gradOutput_t, i * outputWidth * outputHeight)
              for (k <- (0 until outputWidth * outputHeight): Rep[Range]) {
                sum += data(k)
              }
              gradBias.data(i) += scale * sum
            }
        }
      }
    }

    def ConvGradInput(gradOutput: Tensor, gradInput: Tensor, fgradInput: Tensor, weight: Tensor, dH: Int, dW: Int, padH: Int, padW: Int) = {
      val batchSize = gradInput.shape(0)
      val inputHeight = gradInput.shape(2)
      val inputWidth = gradInput.shape(3)
      val nOutputPlane = weight.shape(0)
      val nInputPlane = weight.shape(1)
      val kH = weight.shape(2)
      val kW = weight.shape(3)
      val outputHeight = gradOutput.shape(2)
      val outputWidth = gradOutput.shape(3)
      for (t <- DataLoop(batchSize)) {
        val gradInput_t = gradInput(t).data
        val gradOutput_t = gradOutput(t).data
        val fgradInput_t = fgradInput(t).data
        val dim1 = kW * kH * nInputPlane
        val dim2 = nOutputPlane
        val dim3 = outputHeight * outputWidth
        unchecked[Unit](
          "cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ",
          dim1, ",", dim3, ",", dim2, ",", 1, ",",
          weight.data, ",", dim1, ",", gradOutput_t, ",", dim3, ",", 0, ",", fgradInput_t, ",", dim3, ")")
        unfoldedAcc(fgradInput_t, gradInput_t, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
      }
    }

    def unfoldedAcc(finput: RAF, input: RAF, kW: Rep[Int], kH: Rep[Int], dW: Int, dH: Int, padW: Int, padH: Int, nInputPlane: Rep[Int],
      inputWidth: Rep[Int], inputHeight: Rep[Int], outputWidth: Rep[Int], outputHeight: Rep[Int]) {
      for (nip <- (0 until nInputPlane): Rep[Range]) {
        for (kh <- (0 until kH): Rep[Range]) {
          for (kw <- (0 until kW): Rep[Range]) {
            val src = slice(finput, nip*kH*kW*outputHeight*outputWidth + kh*kW*outputHeight*outputWidth + kw*outputHeight*outputWidth)
            val dst = slice(input, nip*inputHeight*inputWidth)
            if (padW > 0 || padH > 0) {
              for (y <- (0 until outputHeight): Rep[Range]) {
                val iy: Rep[Int] = y * dH - padH + kh
                __ifThenElse ((iy < 0 || iy >= inputHeight), (), {
                  if (dW == 1) {
                    val ix: Rep[Int] = 0 - padW + kw
                    val lpad: Rep[Int] = __ifThenElse((padW-kw > 0), padW-kw, 0)
                    val rpad: Rep[Int] = __ifThenElse((padW-(kW-kw-1) > 0), padW-(kW-kw-1), 0)
                    val dst_slice = slice(dst, iy*inputWidth+ix+lpad)
                    val src_slice = slice(src, y*outputWidth+lpad)
                    for (i <- 0 until (outputWidth - lpad - rpad)) dst_slice(i) += src_slice(i)
                  } else {
                    for (x <- (0 until outputWidth): Rep[Range]) {
                      val ix = x*dW - padW + kw
                      __ifThenElse ((ix < 0 || ix >= inputWidth), (), dst(iy*inputWidth+ix) += src(y*outputWidth+x))
                    }
                  }
                  ()
                })
              }
            } else {
              for (y <- (0 until outputHeight): Rep[Range]) {
                val iy = y*dH + kh
                val ix = kw
                if (dW == 1) {
                  val dst_slice = slice(dst, iy*inputWidth+ix)
                  val src_slice = slice(src, y*outputWidth)
                  for (i <- (0 until outputWidth): Rep[Range]) dst_slice(i) += src_slice(i)
                } else {
                  for (x <- (0 until outputWidth): Rep[Range]) {
                    dst(iy*inputWidth+ix+x*dW) += src(y*outputWidth+x)
                  }
                }
              }
            }
          }
        }
      }
    }

    @virtualize
    override def mask4D(input: Tensor, lengths: Rep[Array[Int]]): Tensor = {
      // inplace mask (input is of size Batch * c * d * Time, lengths are the actual length of each sequence in batch)
      assert (input.rank == 4, s"input of mask function must be 4D, got ${input.shape}")
      for (i <- DataLoop(input.shape(0))) {
        for (j <- DataLoop(input.shape(1))) {
          for (k <- DataLoop(input.shape(2))) {
            for (t <- DataLoop(input.shape(3))) {
              if (t >= lengths(i)) input.data(i * input.shape.strides(0) + j * input.shape.strides(1) + k * input.shape.strides(2) + t) = 0
            }
          }
        }
      }
      input
    }

    @virtualize
    override def relu(x: Tensor, inPlace: Boolean = false): Tensor = {
      val res = if (inPlace) x.data else mallocArray[Float](x.scalarCount)
      for (i <- 0 until x.scalarCount: Rep[Range]) {
        if (x.data(i) < 0.0f)
          res(i) = 0.0f
        else
          res(i) = x.data(i)
      }
      Tensor(res, x.shape.seq : _*)
    }

    @virtualize
    override def relu_grad(input: TensorR, res: TensorR, inPlace: Boolean = false): Unit = {
      for (i <- 0 until input.x.scalarCount: Rep[Range]) {
        if (inPlace) {
          if (input.x.data(i) < 0.0f) input.d.data(i) = 0.0f
        } else {
          input.d.data(i) += (if (input.x.data(i) < 0.0f) 0.0f else res.d.data(i))
        }
      }
    }

    @virtualize
    override def hardTanh(x: Tensor, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Tensor = {
      val res = if (inPlace) x.data else mallocArray[Float](x.scalarCount)
      for (i <- 0 until x.scalarCount: Rep[Range]) {
        if (x.data(i) < min_val) res(i) = min_val
        if (x.data(i) > max_val) res(i) = max_val
      }
      Tensor(res, x.shape.seq: _*)
    }

    @virtualize
    override def hardTanh_grad(input: TensorR, res: TensorR, min_val: Float = -1.0f, max_val: Float = 1.0f, inPlace: Boolean = false): Unit = {
      for (i <- 0 until input.x.scalarCount: Rep[Range]) {
        if (inPlace) {
          if (input.x.data(i) < min_val || input.x.data(i) > max_val) input.d.data(i) = 0.0f
        } else {
          input.d.data(i) += (if (input.x.data(i) < min_val || input.x.data(i) > max_val) 0.0f else res.d.data(i))
        }
      }
    }

    override def tanh(x: Tensor) = x.map(s => Math.tanh(s).toFloat)
    override def tanh_grad(input: TensorR, res: TensorR): Unit = {
      input.d.add_oneMinusSquare_mult(res.x, res.d)
    }

    override def sigmoid(x: Tensor) = x.map(s => 1.0f / (Math.exp(-1.0f * s).toFloat + 1.0f))
    override def sigmoid_grad(input: TensorR, res: TensorR): Unit = {
      input.d.add_oneMinusThenMult_mult(res.x, res.d)
    }

    def buildTensor(dims: Seq[Rep[Int]], byIndex: Rep[Int] => Rep[Float]): Tensor = {
      val res = this.mallocArray[Float](dims.product1)
      for (i <- DataLoop(dims.product1)) res(i) = byIndex(i)
      Tensor(res, dims: _*)
    }

    override def exp(x: Tensor) = buildTensor(x.shape, i => Math.exp(x.data(i)).toFloat)
    override def exp_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) * y.x.data(i) }

    override def log(x: Tensor) = buildTensor(x.shape, i => Math.log(x.data(i)).toFloat)
    override def log_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) / x.x.data(i) }

    override def sqrt(x: Tensor) = buildTensor(x.shape, i => Math.sqrt(x.data(i)).toFloat)
    override def sqrt_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) / y.x.data(i) / 2.0f }

    override def square(x: Tensor) = buildTensor(x.shape, {i => val t = x.data(i); t * t})
    override def square_grad(x: TensorR, y: TensorR): Unit = x.d.mutate { (i: Rep[Int]) => y.d.data(i) * x.x.data(i) * 2.0f }

    @virtualize
    override def softmax(x: Tensor, dim: Int = 1): Tensor = {
      assert(x.rank == 2, s"TODO: Fei Wang, Softmax input must be 2-D: [batchSize, logits] for now, got ${x.shape}")
      assert(dim == 1, s"TODO: Fei Wang, dim must be 1 for now, got ${dim}")
      val max = x.max2D(dim = 1)
      val res = Tensor.zeros_like(x)
      val offset = var_new(0)
      for (batch <- DataLoop(x.shape(0))) {
        for (i <- DataLoop(x.shape(1))) {
          res.data(offset) = Math.exp(x.data(offset) - max.data(batch)).toFloat
          offset += 1
        }
      }
      val sum = res.sum(dim = 1)
      offset = 0
      for (batch <- DataLoop(res.shape(0))) {
        for (i <- DataLoop(res.shape(1))) {
          res.data(offset) = res.data(offset) / sum.data(batch)
          offset += 1
        }
      }
      res
    }

    @virtualize
    override def logSoftmax(x: Tensor, dim: Int = 1): Tensor = {
      assert(x.rank == 2, s"TODO: Fei Wang, Softmax input must be 2-D: [batchSize, logits] for now, got ${x.shape}")
      assert(dim == 1, s"TODO: Fei Wang, dim must be 1 for now, got ${dim}")

      val max = x.max2D(dim = 1)
      val res = Tensor.zeros_like(x)
      // fill res with exp(x_i - max)
      val offset = var_new(0)
      for (batch <- DataLoop(x.shape(0))) {
        for (i <- DataLoop(x.shape(1))) {
          res.data(offset) = Math.exp(x.data(offset) - max.data(batch)).toFloat
          offset += 1
        }
      }
      val sum = res.sum(dim = 1)
      offset = 0
      for (batch <- DataLoop(res.shape(0))) {
        val logsum = max.data(batch) + Math.log(sum.data(batch)).toFloat
        for (i <- DataLoop(res.shape(1))) {
          res.data(offset) = x.data(offset) - logsum
          offset += 1
        }
      }
      res
    }

    // TODO: Implement `softmax_grad` for CPU.
    override def softmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit = ???

    override def logSoftmax_grad(input: TensorR, res: TensorR, dim: Int = 1): Unit = {
      val sum = res.d.sum(dim = 1)
      val offset = var_new(0)
      for (batch <- DataLoop(input.x.shape(0))) {
        for (i <- DataLoop(input.x.shape(1))) {
          input.d.data(offset) += res.d.data(offset) - Math.exp(res.x.data(offset)).toFloat * sum.data(batch)
          offset += 1
        }
      }
    }

    override def maxPool2D_batch(input: Tensor, kernels: Seq[Int], strides: Seq[Int], pads: Option[Seq[Int]] = None): (Tensor, Option[Rep[Array[Int]]]) = {
      assert(input.rank == 4, "the input for maxPool (with batch) should have 4 dimensions")
      assert(kernels.size == 2 && strides.size == 2, "kernels and strides should be size 2")
      pads match {
        case None => ()
        case Some(paddings) => assert(paddings.size == 4, "paddings should be size 4 for maxPool_k_batch")
      }
      val (strideRow :: strideCol :: _) = strides.toList
      val (kernelRow :: kernelCol :: _) = kernels.toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = pads match {
        case None => List(0, 0, 0, 0)
        case Some(paddings) => paddings.toList
      }
      assert(strideRow >= 1 && kernelRow >= 1, "kernel width and stride width should be at least 1")
      assert(strideCol >= 1 && kernelCol >= 1, "kernel height and stride height should be at least 1")
      assert(input.shape(2) + 2 * padUp >= kernelRow && input.shape(3) + 2 * padUp >= kernelCol, "Image too small for maxPool_k: " + input.shape + "|" + (kernelRow, kernelCol))
      assert(padUp == padDown && padUp == padLeft && padUp == padRight && padUp >= 0, "pad should be the same")

      val resWidth = convSize(input.shape(2) + padUp + padDown, kernelRow, strideRow)
      val resHeight = convSize(input.shape(3) + padLeft + padRight, kernelCol, strideCol)
      val res = Tensor.fill(Seq(input.shape(0), input.shape(1), resWidth, resHeight), scala.Float.MinValue)
      val savedIdx = NewArray[Int](res.scalarCount)

      for (i <- DataLoop(input.shape(0))) {
        val ptrInput  = slice(input.data, i * input.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        val ptrIdx    = slice(savedIdx, i * res.shape.strides(0))
        val saveIdxBase = i * input.shape.strides(0)
        maxPool_k_inplace(Tensor(ptrInput, input.shape.drop(1): _*),
          kernelRow, kernelCol, strideRow, strideCol, padUp, padDown, padLeft, padRight,
          Tensor(ptrOutput, res.shape.drop(1): _*), ptrIdx, saveIdxBase)
      }
      (res, Some(savedIdx))
    }

    def maxPool_k_inplace(input: Tensor, kernelRow: Int, kernelCol: Int, strideRow: Int, strideCol: Int,
                          padUp: Int, padDown: Int, padLeft: Int, padRight: Int,
                          res: Tensor, savedIdx: Rep[Array[Int]], saveIdxBase: Rep[Int]): Unit = {
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)

      if (padUp == 0) {
        // looping for the output
        val offout = var_new[Int](0)                              // offset of res, by channel
        val offin  = var_new[Int](0)                              // offset of input, by channel
        for (outPane <- DataLoop(res.shape(0))) {
          val offout_1 = var_new[Int](offout)                     // offset of res, built on offout, by row
          val offin_1  = var_new[Int](offin)                      // offset of input, built on offin, by row
          for (outRow <- DataLoop(res.shape(1))) {
            val offout_2 = var_new[Int](offout_1)                 // offset of res, built on offout_1, by col
            val offin_2  = var_new[Int](offin_1)                  // offset of input, built on offin_1, by col
            for (outCol <- DataLoop(res.shape(2))) {

              // looping for the kernel
              val this_index_1 = var_new[Int](offin_2)            // offset of this (input) by row of kernel size
              for (dummy1 <- DataLoop(kernelRow)) {
                val this_index_2 = var_new[Int](this_index_1)     // offset of this (input), built on this_index_1, by col of kernel size
                for (dummy <- DataLoop(kernelCol)) {
                  __ifThenElse ((input.data(this_index_2) > res.data(offout_2)), {
                    res.data(offout_2) = input.data(this_index_2)
                    savedIdx(offout_2) = this_index_2 + saveIdxBase
                  }, ())
                  this_index_2 += 1
                }
                this_index_1 += input.shape.strides(1)
              }

              offout_2 += 1
              offin_2  += strideCol
            }
            offout_1 += res.shape.strides(1)
            offin_1  += strideRow * input.shape.strides(1)
          }
          offout += res.shape.strides(0)
          offin  += input.shape.strides(0)
        }
      } else {
        // looping for the output
        for (resPane <- DataLoop(res.shape(0))) {
          for (resRow <- DataLoop(res.shape(1))) {
            for (resCol <- DataLoop(res.shape(2))) {
              val resOff = resPane * res.shape.strides(0) + resRow * res.shape.strides(1) + resCol
              // looping for the kernel
              for (kRow <- DataLoop(kernelRow)) {
                for (kCol <- DataLoop(kernelCol)) {
                  val inRow = resRow * strideRow - padUp + kRow
                  val inCol = resCol * strideCol - padUp + kCol
                  __ifThenElse ((inRow < 0 || inRow >= input.shape(1) || inCol < 0 || inCol >= input.shape(2)), (), {
                    val inOff = resPane * input.shape.strides(0) + inRow * input.shape.strides(1) + inCol
                    __ifThenElse ((input.data(inOff) > res.data(resOff)), {
                      res.data(resOff) = input.data(inOff)
                      savedIdx(resOff) = inOff
                    }, ())
                  })
                }
              }
            }
          }
        }
      }
    }

    override def maxPool2D_batch_grad(input: TensorR, output: TensorR, sidx: Option[Rep[Array[Int]]],
                                      kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      sidx match {
        case None => ???
        case Some(sidx) =>
          for (i <- DataLoop(output.d.scalarCount)) {
            input.d.data(sidx(i)) += output.d.data(i)
          }
      }
    }

    override def averagePool2D_batch(input: Tensor, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Tensor = {
      val (strideRow :: strideCol :: Nil) = strides.toList
      val (kernelRow :: kernelCol :: Nil) = kernel.toList
      val (padUp :: padDown :: padLeft :: padRight :: Nil) = pads.toList

      val resWidth = convSize(input.shape(2) + padUp + padDown, kernelRow, strideRow)
      val resHeight = convSize(input.shape(3) + padLeft + padRight, kernelCol, strideCol)
      val res = Tensor.zeros(input.shape(0), input.shape(1), resWidth, resHeight)

      for (i <- DataLoop(input.shape(0))) {
        val ptrInput = slice(input.data, i * input.shape.strides(0))
        val ptrOutput = slice(res.data, i * res.shape.strides(0))
        this.averagePool_inplace(Tensor(ptrInput, input.shape.drop(1): _*),
          kernelRow, kernelCol, strideRow, strideCol, padUp, padDown, padLeft, padRight, Tensor(ptrOutput, res.shape.drop(1): _*))
      }
      res
    }

    @virtualize
    def averagePool_inplace(input: Tensor, kernelRow: Int, kernelCol: Int, strideRow: Int, strideCol: Int, padUp: Int, padDown: Int, padLeft: Int, padRight: Int, res: Tensor): Unit = {
      val resWidth = res.shape(1)
      val resHeight = res.shape(2)
      val kernelSize = kernelRow * kernelCol * 1.0f

      if (padUp == 0) {
        // looping for the output
        for (resPane <- DataLoop(res.shape(0))) {
          for (resRow <- DataLoop(res.shape(1))) {
            for (resCol <- DataLoop(res.shape(2))) {
              val resOff = resPane * res.shape.strides(0) + resRow * res.shape.strides(1) + resCol
              val inOff = resPane * input.shape.strides(0) + resRow * strideRow * input.shape.strides(1) + resCol * strideCol
              // looping for the kernel
              val sum = var_new[Float](0.0f)
              for (kRow <- DataLoop(kernelRow)) {
                for (kCol <- DataLoop(kernelCol)) {
                  sum += input.data(inOff + kRow * input.shape.strides(1) + kCol)
                }
              }
              res.data(resOff) = sum / kernelSize
            }
          }
        }
      } else {
        ???
      }
    }

    override def averagePool2D_batch_grad(input: TensorR, output: TensorR, kernel: Seq[Int], strides: Seq[Int], pads: Seq[Int]): Unit = {
      val strideRow = strides.head
      val strideCol = strides.last
      val kernelRow = kernel.head
      val kernelCol = kernel.last
      val kernelSize = kernelRow * kernelCol
      val pad = pads(0)

      if (pad == 0) {
        for (batch <- DataLoop(input.x.shape(0))) {
          // looping for the output
          for (yPane <- DataLoop(output.x.shape(1))) {
            for (yRow <- DataLoop(output.x.shape(2))) {
              for (yCol <- DataLoop(output.x.shape(3))) {
                val indexCurr = batch * output.x.shape.strides(0) + yPane * output.x.shape.strides(1) + yRow * output.x.shape.strides(2) + yCol
                val dCurr = output.d.data(indexCurr) / kernelSize
                val indexThis = batch * input.x.shape.strides(0) + yPane * input.x.shape.strides(1) + yRow * strideRow * input.x.shape.strides(2) + yCol * strideCol
                // looping for the kernel
                for (kRow <- DataLoop(kernelRow)) {
                  for (kCol <- DataLoop(kernelCol)) {
                    input.d.data(indexThis + kRow * input.x.shape.strides(2) + kCol) += dCurr
                  }
                }
              }
            }
          }
        }
      } else {
        ???
      }
    }

    override def batchNormInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = {
      val epsilon: Float = 0.00001f
      val out1 = (x - runningMean.resize(1,-1,1,1)) / (runningVar + epsilon).sqrt().resize(1, -1, 1, 1)
      val res = out1 * scale.resize(1,-1,1,1) + bias.resize(1,-1,1,1)
      res
    }

    override def batchNormTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor]) = {
      val saveMean = x.batchNormAv()
      val diff = x - saveMean
      val saveInvVariance = diff.square().batchNormAv()
      val epsilon = 0.00001f
      val xhat = diff / (saveInvVariance + epsilon).sqrt()
      val outy = xhat * scale.resize(-1, 1, 1) + bias.resize(-1, 1, 1)
      // runningMean and runningVariance should also be updated???
      (outy, Some(saveMean), Some(saveInvVariance))
    }

    override def batchNorm_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor]): Unit = {
      ???
    }

    override def batchNorm1DInference(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): Tensor = ???
    override def batchNorm1DTraining(x: Tensor, scale: Tensor, bias: Tensor, runningMean: Tensor, runningVar: Tensor): (Tensor, Option[Tensor], Option[Tensor]) = ???
    override def batchNorm1D_grad(input: TensorR, res: TensorR, scale: TensorR, bias: TensorR, saveMean: Option[Tensor], saveInvVariance: Option[Tensor]): Unit = ???

    @virtualize
    override def dropout(input: Tensor, prob: Float = 0.5f): (Tensor, Rep[Array[Float]], Rep[Int]) = {
      assert(0.0f <= prob && prob < 1.0f, s"dropout rate should be [0.0, 1), got $prob")

      val res = backend.mallocArray[Float](input.scalarCount)
      val mask = backend.mallocArray[Float](input.scalarCount)
      val scale = 1.0f / (1.0f - prob)

      for (i <- DataLoop(input.scalarCount)) {
        if (Random.rand() > prob) {
          res(i) = input.data(i) * scale
          mask(i) = scale
        } else {
          res(i) = 0.0f
          mask(i) = 0.0f
        }
      }
      (Tensor(res, input.shape.seq : _*), mask, 0)
    }

    override def dropout_grad(input: TensorR, output: TensorR, prob: Float, helper: Rep[Array[Float]], size: Rep[Int]): Unit = {
      input.d += Tensor(helper, input.x.shape: _*) * output.d  // TODO (Fei Wang): should optimized by fusing loops
    }

    override def nllLoss(x: Tensor, target: Rep[Array[Int]]): Tensor = {
      assert(x.rank == 2, "Input must be a 2-D tensor")
      generateRawComment("nllLoss forward in CPU")
      val batchSize = x.shape(0)
      val res = mallocArray[Float](batchSize)
      val offset = var_new(0)
      for (batch <- DataLoop(batchSize)) {
        res(batch) = -1.0f * x.data(offset + target(batch))
        offset += x.shape.strides(0)
      }
      Tensor(res, batchSize)
    }

    override def nllLoss_grad(input: TensorR, res: TensorR, target: Rep[Array[Int]]): Unit = {
      generateRawComment("nllLoss_grad implementation in CPU")
      val offset = var_new(0)
      for (batch <- DataLoop(input.d.shape(0))) {
        input.d.data(offset + target(batch)) += -1.0f * res.d.data(batch)
        offset += input.d.shape.strides(0)
      }
    }

    // CTCLoss
    override def ctcLoss(prob: TensorR, inputLengths: Rep[Array[Int]], labels: Rep[Array[Int]], labelLengths: Rep[Array[Int]]): Tensor = ???

    override def sum(x: Tensor): Tensor = {
      Tensor.scalar(x.fold(0.0f)(_ + _))
    }
    override def sum_grad(input: TensorR, res: TensorR): Unit = { +=(input.d, res.d) }
    override def mean(x: Tensor): Tensor = {
      this.sum(x) / x.scalarCount
    }
    override def mean_grad(input: TensorR, res: TensorR): Unit = {
      += (input.d, res.d / input.x.scalarCount)  // TODO (Fei Wang): optimize
    }
    override def sum(input: Tensor, dim: Int) = {
      assert(dim >= 0 && dim < input.rank, "dim should be within range of this.nbDims")
      val higherDims = input.shape.take(dim)
      val higherDimsSquashed = higherDims.product1
      val resDims = higherDims ++ input.shape.drop(dim + 1)
      val res = Tensor.zeros(resDims: _*)

      // looping over the dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the dimension to be summed
        val offres = var_new(high * (if (dim == 0) res.scalarCount else res.shape.strides(dim - 1)))
        val offthis = var_new(high * (if (dim == 0) input.scalarCount else input.shape.strides(dim - 1)))
        for (sum <- DataLoop(input.shape(dim))) {
          // looping over the dims lower than dim
          for (low <- DataLoop(input.shape.strides(dim))) {
            res.data(offres + low) += input.data(offthis + low)
          }
          offthis += input.shape.strides(dim)
        }
      }
      res
    }
    override def sum_grad(input: TensorR, output: TensorR, dim: Int): Unit = {
      val higherDims = input.x.shape.take(dim)
      val higherDimsSquashed = higherDims.product1
      val resDims = higherDims ++ input.x.shape.drop(dim + 1)
      // looping over the dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the dimension to be summed
        val offres = var_new(high * (if (dim == 0) output.x.scalarCount else output.x.shape.strides(dim - 1)))
        val offthis = var_new(high * (if (dim == 0) input.x.scalarCount else input.x.shape.strides(dim - 1)))
        for (sum <- DataLoop(input.x.shape(dim))) {
          // looping over the dims lower than dim
          for (low <- DataLoop(input.x.shape.strides(dim))) {
            input.d.data(offthis + low) += output.d.data(offres + low)
          }
          offthis += input.x.shape.strides(dim)
        }
      }
    }

    override def concat(dim: Int, tensors: Seq[Tensor]): Tensor = {
      // prepare result tensor
      val higherDims = tensors(0).shape.take(dim)
      val higherDimsSquashed = higherDims.product1
      val resDims    = (0 until tensors(0).rank: Range).map { i =>
        if (i != dim) tensors(0).shape(i)
        else tensors.map(x => x.shape(dim)).sum1
      }
      val totalnbElem = resDims.product1

      val res = this.mallocArray[Float](totalnbElem)
      val targetId = var_new(0)             // this is the index of res to write to
      // looping over dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the concatenation dim
        for (whichTensor <- tensors) {
          // looping over the dimensions lower than or equal to dim, in the current tensor
          val stride = if (dim == 0) whichTensor.shape.scalarCount else whichTensor.shape.strides(dim-1)
          val ptrIntput = slice(whichTensor.data, high * stride)
          for (lowOrEqual <- DataLoop(stride)) {
            res(targetId) = ptrIntput(lowOrEqual)
            targetId += 1
          }
        }
      }
      Tensor(res, resDims: _*)
    }

    override def concat_grad(dim: Int, tensorRs: Seq[TensorR], output: TensorR): Unit = {
      val higherDims = tensorRs(0).x.shape.take(dim)
      val higherDimsSquashed = higherDims.product1

      val targetId = var_new(0)        // this is the index of res to read gradient from
      // looping over dims higher than dim, squashed
      for (high <- DataLoop(higherDimsSquashed)) {
        // looping over the concatenation dim
        for (whichTensorR <- tensorRs) {
          // looping over the dimensions lower than or equal to dim (but within an input tensor)
          val stride = if (dim == 0) whichTensorR.x.shape.scalarCount else whichTensorR.x.shape.strides(dim-1)
          val ptrInput = slice(whichTensorR.d.data, high * stride)
          for (lowOrEqual <- DataLoop(stride)) {
            ptrInput(lowOrEqual) += output.d.data(targetId)
            targetId += 1
          }
        }
      }
    }

    override def repeat0(in: Tensor, context: Int): Tensor = ???
    override def repeat0_grad(in: TensorR, out: TensorR, context: Int): Unit = ???

    @virtualize
    override def adagrad_update(tr: TensorR, t: Tensor, learning_rate: Float, gradClip: Float, descent: Boolean): Unit = {
      tr.d.changeTo { i =>
        val temp = var_new(tr.d.data(i))
        if (temp > gradClip) temp = gradClip
        if (temp < -gradClip) temp = -gradClip
        t.data(i) += temp * temp
        if (descent)
          tr.x.data(i) -= learning_rate * temp / Math.sqrt(t.data(i) + 1e-8f).toFloat
        else
          tr.x.data(i) += learning_rate * temp / Math.sqrt(t.data(i) + 1e-8f).toFloat
        0.0f
      }
    }

    @virtualize
    override def momentum_update(tr: TensorR, t: Tensor, learning_rate: Float, momentum: Float, gradClip: Float, nesterov: Boolean, descent: Boolean): Unit = {
      tr.d.changeTo { i =>
        val temp = var_new(tr.d.data(i))
        if (temp > gradClip) temp = gradClip
        if (temp < -gradClip) temp = -gradClip
        t.data(i) *= momentum
        t.data(i) += temp
        if (nesterov) { temp += momentum * t.data(i) }
        else { temp = t.data(i) }
        if (descent) { tr.x.data(i) -= learning_rate * temp }
        else { tr.x.data(i) += learning_rate * temp }
        0.0f
      }
    }

  }

  object BackendCPU {
    def apply() = new BackendCPU
  }

  // The current backend for code generation.
  // To switch code generation to a different backend, simply change this value
  // in your DSL program.
  var backend: Backend = BackendCPU()

}
