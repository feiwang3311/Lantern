package lantern

import lms.core.stub._
import lms.core.virtualize
import lms.macros.SourceContext

class TestCudnn extends LanternFunSuite {

  testGPU("broadCastingPlus1") {
    val plus = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2, 3), 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
        val tensor2 = Tensor.fromData(Seq(2, 3), 6,5,4,3,2,1)
        Tensor.assertEqual((tensor1 + tensor2).toCPU(), Tensor(Array[Float](7,7,7,7,7,7), 2, 3))
        Tensor.assertEqual((tensor2 + tensor1).toCPU(), Tensor(Array[Float](7,7,7,7,7,7), 2, 3))

	      // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
      }
    }
    runTest(plus)
  }

  testGPU("broadCastingPlus2") {
    val plus2 = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2, 3), 1,2,3,4,5,6)
        val tensor2 = Tensor.fromData(Seq(2, 1), 1,2)
        Tensor.assertEqual((tensor1 + tensor2).toCPU(), Tensor(Array[Float](2,3,4,6,7,8), 2, 3))
        Tensor.assertEqual((tensor2 + tensor1).toCPU(), Tensor(Array[Float](2,3,4,6,7,8), 2, 3))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](3, 3), 2, 1))
      }
    }
    runTest(plus2)
  }

  testGPU("broadCastingPlus3") {
    val test3 = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2, 3), 1,2,3,4,5,6)
        val tensor2 = Tensor.fromData(Seq(1, 3), 3,4,5)
        Tensor.assertEqual((tensor1 + tensor2).toCPU(), Tensor(Array[Float](4,6,8,7,9,11), 2,3))
        Tensor.assertEqual((tensor2 + tensor1).toCPU(), Tensor(Array[Float](4,6,8,7,9,11), 2,3))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,1,1,1,1,1), 2, 3))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](2, 2, 2), 1, 3))
      }
    }
    runTest(test3)
  }

  testGPU("broadCastingPlus4") {
    val test4 = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2, 2, 2), 1,2,3,4,5,6,7,8)
        val tensor2 = Tensor.fromData(Seq(2), 1,2)
        Tensor.assertEqual((tensor1 + tensor2).toCPU(), Tensor(Array[Float](2,4,4,6,6,8,8,10), 2,2,2))
        Tensor.assertEqual((tensor2 + tensor1).toCPU(), Tensor(Array[Float](2,4,4,6,6,8,8,10), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,1,1,1,1,1,1,1), 2, 2, 2))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](4, 4), 2))
      }
    }
    runTest(test4)
  }

  testGPU("broadCastingPlus5") {
    val test5 = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2,2,2), 1,2,3,4,5,6,7,8)
        val tensor2 = Tensor.fromData(Seq(2,1,2), 1,2,3,4)
        Tensor.assertEqual((tensor1 + tensor2).toCPU(), Tensor(Array[Float](2,4,4,6,8,10,10,12), 2,2,2))
        Tensor.assertEqual((tensor2 + tensor1).toCPU(), Tensor(Array[Float](2,4,4,6,8,10,10,12), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a + b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,1,1,1,1,1,1,1), 2, 2, 2))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](2,2,2,2), 2, 1, 2))
      }
    }
    runTest(test5)
  }

  testGPU("broadCastingMinus5") {
    val test5 = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2,2,2), 1,2,3,4,5,6,7,8)
        val tensor2 = Tensor.fromData(Seq(2,1,2), 1,2,3,4)
        Tensor.assertEqual((tensor1 - tensor2).toCPU(), Tensor(Array[Float](0,0,2,2,2,2,4,4), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a - b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,1,1,1,1,1,1,1), 2, 2, 2))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](-2,-2,-2,-2), 2, 1, 2))
      }
    }
    runTest(test5)
  }

  testGPU("broadCastingMult5") {
    val test5 = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2,2,2), 1,2,3,4,5,6,7,8)
        val tensor2 = Tensor.fromData(Seq(2,1,2), 1,2,3,4)
        Tensor.assertEqual((tensor1 * tensor2).toCPU(), Tensor(Array[Float](1,4,3,8,15,24,21,32), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a * b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,2,1,2,3,4,3,4), 2, 2, 2))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](4,6,12,14), 2, 1, 2))
      }
    }
    runTest(test5)
  }

  testGPU("broadCastingDiv5") {
    val test5 = new LanternDriverCudnn[String, Unit] {
      override val fileName = currentTestName
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val tensor1 = Tensor.fromData(Seq(2,2,2), 1,2,3,4,5,6,7,8)
        val tensor2 = Tensor.fromData(Seq(2,1,2), 1,2,2,4)
        Tensor.assertEqual((tensor1 / tensor2).toCPU(), Tensor(Array[Float](1,1,3,2,2.5f,1.5f,3.5f,2), 2,2,2))

        // test backprop
        val a = TensorR(tensor1)
        val b = TensorR(tensor2)
        def loss(dummy: TensorR) = (a / b).sum()
        gradR_loss(loss)(Tensor.zeros(1))
        Tensor.assertEqual(a.d.toCPU(), Tensor(Array[Float](1,0.5f,1,0.5f,0.5f,0.25f,0.5f,0.25f), 2, 2, 2))
        Tensor.assertEqual(b.d.toCPU(), Tensor(Array[Float](-4, -1.5f, -3, -0.875f), 2, 1, 2))
      }
    }
    runTest(test5)
  }

  testGPU("permute0") {
    val permute = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-permute0"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(4, 3), -1,2,-3,-4,5,6,-7,8,-9,-10,11,-12)
        val result1 = x.permute(1, 0)
        val grad1 = gradR(x => x.permute(1, 0).relu(false))(x)

        backend = BackendCPU()
        val expected1 = Tensor.fromData(Seq(3, 4), -1, -4, -7, -10, 2, 5, 8, 11, -3, 6, -9, -12)
        val expectedGrad1 = Tensor.fromData(Seq(4, 3), 0,1,0,0,1,1, 0,1,0, 0,1,0)
        Tensor.assertEqual(expected1, result1.toCPU())
        Tensor.assertEqual(expectedGrad1, grad1.toCPU())
      }
    }
    runTest(permute)
  }

  testGPU("permute1") {
    val permute = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-permute1"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(2, 2, 3), -1,2,-3,-4,5,6,-7,8,-9,-10,11,-12)
        val result1 = x.permute(0, 2, 1)
        val result2 = x.permute(1, 0, 2)
        val result3 = x.permute(2, 0, 1)
        val result4 = x.permute(1, 2, 0)
        val result5 = x.permute(2, 1, 0)
        val grad1 = gradR(x => x.permute(0, 2, 1).relu(false))(x)
        val grad2 = gradR(x => x.permute(1, 0, 2).relu(true))(x)
        val grad3 = gradR(x => x.permute(2, 0, 1).relu(false))(x)
        val grad4 = gradR(x => x.permute(1, 2, 0).relu(true))(x)
        val grad5 = gradR(x => x.permute(2, 1, 0).relu(false))(x)

        backend = BackendCPU()
        val expected1 = Tensor.fromData(Seq(2, 3, 2), -1,-4,2,5,-3,6, -7,-10,8,11,-9,-12)
        val expected2 = Tensor.fromData(Seq(2, 2, 3), -1,2,-3,  -7,8,-9, -4,5,6, -10,11,-12)
        val expected3 = Tensor.fromData(Seq(3, 2, 2), -1, -4, -7, -10, 2, 5, 8, 11, -3, 6,-9, -12)
        val expected4 = Tensor.fromData(Seq(2, 3, 2), -1,  -7,   2,   8,  -3,  -9,  -4, -10,   5,  11,   6, -12)
        val expected5 = Tensor.fromData(Seq(3, 2, 2), -1,  -7,  -4, -10,   2,   8,   5,  11,  -3,  -9,   6, -12)

        val expectedGrad = Tensor.fromData(Seq(2, 2, 3), 0,1,0,0,1,1, 0,1,0, 0,1,0)
        Tensor.assertEqual(expected1, result1.toCPU())
        Tensor.assertEqual(expected2, result2.toCPU())
        Tensor.assertEqual(expected3, result3.toCPU())
        Tensor.assertEqual(expected4, result4.toCPU())
        Tensor.assertEqual(expected5, result5.toCPU())
        Tensor.assertEqual(expectedGrad, grad1.toCPU())
        Tensor.assertEqual(expectedGrad, grad2.toCPU())
        Tensor.assertEqual(expectedGrad, grad3.toCPU())
        Tensor.assertEqual(expectedGrad, grad4.toCPU())
        Tensor.assertEqual(expectedGrad, grad5.toCPU())
      }
    }
    runTest(permute)
  }

  /*
  testGPU("permute2") {
    val permute = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-permute"
      @virtualize
      def snippet(x: Rep[String]): Rep[Unit] = {
        val x = Tensor.fromData(Seq(3, 4, 4, 3),
            0,    1,    2,    3,    4,   -5,   -6,   -7,    8,    9,   10,
          11,  -12,  -13,  -14,   15,   16,   17,  -18,  -19,  -20,  -21,
          22,   23,   24,   25,  -26,   27,  -28,   29,   30,   31,   32,
          33,   34,   35,  -36,  -37,  -38,   39,   40,   41,  -42,   43,
          -44,  -45,   46,  -47,   48,   49,   50,   51,   52,  -53,  -54,
          -55,   56,  -57,   58,   59,  -60,   61,  -62,   63,   64,  -65,
          -66,  -67,  -68,  -69,   70,  -71,  -72,   73,  -74,  -75,   76,
          -77,   78,   79,   80,  -81,  -82,  -83,  -84,   85,   86,   87,
          88,  -89,  -90,  -91,  -92,  -93,  -94,   95,   96,  -97,  -98,
          99, -100, -101,  102,  103, -104, -105,  106, -107,  108,  109,
          -110, -111, -112, -113,  114, -115,  116,  117,  118, -119, -120,
          -121,  122,  123, -124,  125,  126, -127,  128, -129,  130, -131,
          -132,  133, -134, -135, -136, -137, -138, -139,  140, -141,  142,
          -143)
        val perms = List(0,1,2,3).permutations.toList.drop(1)
        val resGrad = perms map { perm =>
          val result = x.permute(perm:_*)
          val grad = gradR(x => x.permute(perm:_*).relu(true))(x)
          (result, grad)
        }

        backend = BackendCPU()
        val expects = Seq(
          Tensor.fromData(Seq(3, 4, 3, 4),
          0,    3,   -6,    9,    1,    4,   -7,   10,    2,   -5,    8,
         11,  -12,   15,  -18,  -21,  -13,   16,  -19,   22,  -14,   17,
        -20,   23,   24,   27,   30,   33,   25,  -28,   31,   34,  -26,
         29,   32,   35,  -36,   39,  -42,  -45,  -37,   40,   43,   46,
        -38,   41,  -44,  -47,   48,   51,  -54,  -57,   49,   52,  -55,
         58,   50,  -53,   56,   59,  -60,   63,  -66,  -69,   61,   64,
        -67,   70,  -62,  -65,  -68,  -71,  -72,  -75,   78,  -81,   73,
         76,   79,  -82,  -74,  -77,   80,  -83,  -84,   87,  -90,  -93,
         85,   88,  -91,  -94,   86,  -89,  -92,   95,   96,   99,  102,
       -105,  -97, -100,  103,  106,  -98, -101, -104, -107,  108, -111,
        114,  117,  109, -112, -115,  118, -110, -113,  116, -119, -120,
        123,  126, -129, -121, -124, -127,  130,  122,  125,  128, -131,
       -132, -135, -138, -141,  133, -136, -139,  142, -134, -137,  140,
       -143),
          Tensor.fromData(Seq(3, 4, 4, 3),
          0,    1,    2,  -12,  -13,  -14,   24,   25,  -26,  -36,  -37,
        -38,    3,    4,   -5,   15,   16,   17,   27,  -28,   29,   39,
         40,   41,   -6,   -7,    8,  -18,  -19,  -20,   30,   31,   32,
        -42,   43,  -44,    9,   10,   11,  -21,   22,   23,   33,   34,
         35,  -45,   46,  -47,   48,   49,   50,  -60,   61,  -62,  -72,
         73,  -74,  -84,   85,   86,   51,   52,  -53,   63,   64,  -65,
        -75,   76,  -77,   87,   88,  -89,  -54,  -55,   56,  -66,  -67,
        -68,   78,   79,   80,  -90,  -91,  -92,  -57,   58,   59,  -69,
         70,  -71,  -81,  -82,  -83,  -93,  -94,   95,   96,  -97,  -98,
        108,  109, -110, -120, -121,  122, -132,  133, -134,   99, -100,
       -101, -111, -112, -113,  123, -124,  125, -135, -136, -137,  102,
        103, -104,  114, -115,  116,  126, -127,  128, -138, -139,  140,
       -105,  106, -107,  117,  118, -119, -129,  130, -131, -141,  142,
       -143),
          Tensor.fromData(Seq(3, 4, 3, 4),
           0,  -12,   24,  -36,    1,  -13,   25,  -37,    2,  -14,  -26,
        -38,    3,   15,   27,   39,    4,   16,  -28,   40,   -5,   17,
         29,   41,   -6,  -18,   30,  -42,   -7,  -19,   31,   43,    8,
        -20,   32,  -44,    9,  -21,   33,  -45,   10,   22,   34,   46,
         11,   23,   35,  -47,   48,  -60,  -72,  -84,   49,   61,   73,
         85,   50,  -62,  -74,   86,   51,   63,  -75,   87,   52,   64,
         76,   88,  -53,  -65,  -77,  -89,  -54,  -66,   78,  -90,  -55,
        -67,   79,  -91,   56,  -68,   80,  -92,  -57,  -69,  -81,  -93,
         58,   70,  -82,  -94,   59,  -71,  -83,   95,   96,  108, -120,
       -132,  -97,  109, -121,  133,  -98, -110,  122, -134,   99, -111,
        123, -135, -100, -112, -124, -136, -101, -113,  125, -137,  102,
        114,  126, -138,  103, -115, -127, -139, -104,  116,  128,  140,
       -105,  117, -129, -141,  106,  118,  130,  142, -107, -119, -131,
       -143),
          Tensor.fromData(Seq(3, 3, 4, 4),
            0,    3,   -6,    9,  -12,   15,  -18,  -21,   24,   27,   30,
         33,  -36,   39,  -42,  -45,    1,    4,   -7,   10,  -13,   16,
        -19,   22,   25,  -28,   31,   34,  -37,   40,   43,   46,    2,
         -5,    8,   11,  -14,   17,  -20,   23,  -26,   29,   32,   35,
        -38,   41,  -44,  -47,   48,   51,  -54,  -57,  -60,   63,  -66,
        -69,  -72,  -75,   78,  -81,  -84,   87,  -90,  -93,   49,   52,
        -55,   58,   61,   64,  -67,   70,   73,   76,   79,  -82,   85,
         88,  -91,  -94,   50,  -53,   56,   59,  -62,  -65,  -68,  -71,
        -74,  -77,   80,  -83,   86,  -89,  -92,   95,   96,   99,  102,
       -105,  108, -111,  114,  117, -120,  123,  126, -129, -132, -135,
       -138, -141,  -97, -100,  103,  106,  109, -112, -115,  118, -121,
       -124, -127,  130,  133, -136, -139,  142,  -98, -101, -104, -107,
       -110, -113,  116, -119,  122,  125,  128, -131, -134, -137,  140,
       -143),
          Tensor.fromData(Seq(3, 3, 4, 4),
          0,  -12,   24,  -36,    3,   15,   27,   39,   -6,  -18,   30,
        -42,    9,  -21,   33,  -45,    1,  -13,   25,  -37,    4,   16,
        -28,   40,   -7,  -19,   31,   43,   10,   22,   34,   46,    2,
        -14,  -26,  -38,   -5,   17,   29,   41,    8,  -20,   32,  -44,
         11,   23,   35,  -47,   48,  -60,  -72,  -84,   51,   63,  -75,
         87,  -54,  -66,   78,  -90,  -57,  -69,  -81,  -93,   49,   61,
         73,   85,   52,   64,   76,   88,  -55,  -67,   79,  -91,   58,
         70,  -82,  -94,   50,  -62,  -74,   86,  -53,  -65,  -77,  -89,
         56,  -68,   80,  -92,   59,  -71,  -83,   95,   96,  108, -120,
       -132,   99, -111,  123, -135,  102,  114,  126, -138, -105,  117,
       -129, -141,  -97,  109, -121,  133, -100, -112, -124, -136,  103,
       -115, -127, -139,  106,  118,  130,  142,  -98, -110,  122, -134,
       -101, -113,  125, -137, -104,  116,  128,  140, -107, -119, -131,
       -143),
          Tensor.fromData(Seq(4, 3, 4, 3),
           0,    1,    2,    3,    4,   -5,   -6,   -7,    8,    9,   10,
         11,   48,   49,   50,   51,   52,  -53,  -54,  -55,   56,  -57,
         58,   59,   96,  -97,  -98,   99, -100, -101,  102,  103, -104,
       -105,  106, -107,  -12,  -13,  -14,   15,   16,   17,  -18,  -19,
        -20,  -21,   22,   23,  -60,   61,  -62,   63,   64,  -65,  -66,
        -67,  -68,  -69,   70,  -71,  108,  109, -110, -111, -112, -113,
        114, -115,  116,  117,  118, -119,   24,   25,  -26,   27,  -28,
         29,   30,   31,   32,   33,   34,   35,  -72,   73,  -74,  -75,
         76,  -77,   78,   79,   80,  -81,  -82,  -83, -120, -121,  122,
        123, -124,  125,  126, -127,  128, -129,  130, -131,  -36,  -37,
        -38,   39,   40,   41,  -42,   43,  -44,  -45,   46,  -47,  -84,
         85,   86,   87,   88,  -89,  -90,  -91,  -92,  -93,  -94,   95,
       -132,  133, -134, -135, -136, -137, -138, -139,  140, -141,  142,
       -143),
          Tensor.fromData(Seq(4, 3, 3, 4),
           0,    3,   -6,    9,    1,    4,   -7,   10,    2,   -5,    8,
         11,   48,   51,  -54,  -57,   49,   52,  -55,   58,   50,  -53,
         56,   59,   96,   99,  102, -105,  -97, -100,  103,  106,  -98,
       -101, -104, -107,  -12,   15,  -18,  -21,  -13,   16,  -19,   22,
        -14,   17,  -20,   23,  -60,   63,  -66,  -69,   61,   64,  -67,
         70,  -62,  -65,  -68,  -71,  108, -111,  114,  117,  109, -112,
       -115,  118, -110, -113,  116, -119,   24,   27,   30,   33,   25,
        -28,   31,   34,  -26,   29,   32,   35,  -72,  -75,   78,  -81,
         73,   76,   79,  -82,  -74,  -77,   80,  -83, -120,  123,  126,
       -129, -121, -124, -127,  130,  122,  125,  128, -131,  -36,   39,
        -42,  -45,  -37,   40,   43,   46,  -38,   41,  -44,  -47,  -84,
         87,  -90,  -93,   85,   88,  -91,  -94,   86,  -89,  -92,   95,
       -132, -135, -138, -141,  133, -136, -139,  142, -134, -137,  140,
       -143),
          Tensor.fromData(Seq(4, 4, 3, 3),
          0,    1,    2,   48,   49,   50,   96,  -97,  -98,    3,    4,
         -5,   51,   52,  -53,   99, -100, -101,   -6,   -7,    8,  -54,
        -55,   56,  102,  103, -104,    9,   10,   11,  -57,   58,   59,
       -105,  106, -107,  -12,  -13,  -14,  -60,   61,  -62,  108,  109,
       -110,   15,   16,   17,   63,   64,  -65, -111, -112, -113,  -18,
        -19,  -20,  -66,  -67,  -68,  114, -115,  116,  -21,   22,   23,
        -69,   70,  -71,  117,  118, -119,   24,   25,  -26,  -72,   73,
        -74, -120, -121,  122,   27,  -28,   29,  -75,   76,  -77,  123,
       -124,  125,   30,   31,   32,   78,   79,   80,  126, -127,  128,
         33,   34,   35,  -81,  -82,  -83, -129,  130, -131,  -36,  -37,
        -38,  -84,   85,   86, -132,  133, -134,   39,   40,   41,   87,
         88,  -89, -135, -136, -137,  -42,   43,  -44,  -90,  -91,  -92,
       -138, -139,  140,  -45,   46,  -47,  -93,  -94,   95, -141,  142,
       -143),
          Tensor.fromData(Seq(4, 4, 3, 3),
          0,   48,   96,    1,   49,  -97,    2,   50,  -98,    3,   51,
         99,    4,   52, -100,   -5,  -53, -101,   -6,  -54,  102,   -7,
        -55,  103,    8,   56, -104,    9,  -57, -105,   10,   58,  106,
         11,   59, -107,  -12,  -60,  108,  -13,   61,  109,  -14,  -62,
       -110,   15,   63, -111,   16,   64, -112,   17,  -65, -113,  -18,
        -66,  114,  -19,  -67, -115,  -20,  -68,  116,  -21,  -69,  117,
         22,   70,  118,   23,  -71, -119,   24,  -72, -120,   25,   73,
       -121,  -26,  -74,  122,   27,  -75,  123,  -28,   76, -124,   29,
        -77,  125,   30,   78,  126,   31,   79, -127,   32,   80,  128,
         33,  -81, -129,   34,  -82,  130,   35,  -83, -131,  -36,  -84,
       -132,  -37,   85,  133,  -38,   86, -134,   39,   87, -135,   40,
         88, -136,   41,  -89, -137,  -42,  -90, -138,   43,  -91, -139,
        -44,  -92,  140,  -45,  -93, -141,   46,  -94,  142,  -47,   95,
       -143),
          Tensor.fromData(Seq(4, 3, 3, 4),
          0,    3,   -6,    9,   48,   51,  -54,  -57,   96,   99,  102,
       -105,    1,    4,   -7,   10,   49,   52,  -55,   58,  -97, -100,
        103,  106,    2,   -5,    8,   11,   50,  -53,   56,   59,  -98,
       -101, -104, -107,  -12,   15,  -18,  -21,  -60,   63,  -66,  -69,
        108, -111,  114,  117,  -13,   16,  -19,   22,   61,   64,  -67,
         70,  109, -112, -115,  118,  -14,   17,  -20,   23,  -62,  -65,
        -68,  -71, -110, -113,  116, -119,   24,   27,   30,   33,  -72,
        -75,   78,  -81, -120,  123,  126, -129,   25,  -28,   31,   34,
         73,   76,   79,  -82, -121, -124, -127,  130,  -26,   29,   32,
         35,  -74,  -77,   80,  -83,  122,  125,  128, -131,  -36,   39,
        -42,  -45,  -84,   87,  -90,  -93, -132, -135, -138, -141,  -37,
         40,   43,   46,   85,   88,  -91,  -94,  133, -136, -139,  142,
        -38,   41,  -44,  -47,   86,  -89,  -92,   95, -134, -137,  140,
       -143),
          Tensor.fromData(Seq(4, 3, 4, 3),
          0,   48,   96,    3,   51,   99,   -6,  -54,  102,    9,  -57,
       -105,    1,   49,  -97,    4,   52, -100,   -7,  -55,  103,   10,
         58,  106,    2,   50,  -98,   -5,  -53, -101,    8,   56, -104,
         11,   59, -107,  -12,  -60,  108,   15,   63, -111,  -18,  -66,
        114,  -21,  -69,  117,  -13,   61,  109,   16,   64, -112,  -19,
        -67, -115,   22,   70,  118,  -14,  -62, -110,   17,  -65, -113,
        -20,  -68,  116,   23,  -71, -119,   24,  -72, -120,   27,  -75,
        123,   30,   78,  126,   33,  -81, -129,   25,   73, -121,  -28,
         76, -124,   31,   79, -127,   34,  -82,  130,  -26,  -74,  122,
         29,  -77,  125,   32,   80,  128,   35,  -83, -131,  -36,  -84,
       -132,   39,   87, -135,  -42,  -90, -138,  -45,  -93, -141,  -37,
         85,  133,   40,   88, -136,   43,  -91, -139,   46,  -94,  142,
        -38,   86, -134,   41,  -89, -137,  -44,  -92,  140,  -47,   95,
       -143),
          Tensor.fromData(Seq(4, 3, 4, 3),
          0,    1,    2,  -12,  -13,  -14,   24,   25,  -26,  -36,  -37,
        -38,   48,   49,   50,  -60,   61,  -62,  -72,   73,  -74,  -84,
         85,   86,   96,  -97,  -98,  108,  109, -110, -120, -121,  122,
       -132,  133, -134,    3,    4,   -5,   15,   16,   17,   27,  -28,
         29,   39,   40,   41,   51,   52,  -53,   63,   64,  -65,  -75,
         76,  -77,   87,   88,  -89,   99, -100, -101, -111, -112, -113,
        123, -124,  125, -135, -136, -137,   -6,   -7,    8,  -18,  -19,
        -20,   30,   31,   32,  -42,   43,  -44,  -54,  -55,   56,  -66,
        -67,  -68,   78,   79,   80,  -90,  -91,  -92,  102,  103, -104,
        114, -115,  116,  126, -127,  128, -138, -139,  140,    9,   10,
         11,  -21,   22,   23,   33,   34,   35,  -45,   46,  -47,  -57,
         58,   59,  -69,   70,  -71,  -81,  -82,  -83,  -93,  -94,   95,
       -105,  106, -107,  117,  118, -119, -129,  130, -131, -141,  142,
       -143),
         Tensor.fromData(Seq(4, 3, 3, 4),
           0,  -12,   24,  -36,    1,  -13,   25,  -37,    2,  -14,  -26,
        -38,   48,  -60,  -72,  -84,   49,   61,   73,   85,   50,  -62,
        -74,   86,   96,  108, -120, -132,  -97,  109, -121,  133,  -98,
       -110,  122, -134,    3,   15,   27,   39,    4,   16,  -28,   40,
         -5,   17,   29,   41,   51,   63,  -75,   87,   52,   64,   76,
         88,  -53,  -65,  -77,  -89,   99, -111,  123, -135, -100, -112,
       -124, -136, -101, -113,  125, -137,   -6,  -18,   30,  -42,   -7,
        -19,   31,   43,    8,  -20,   32,  -44,  -54,  -66,   78,  -90,
        -55,  -67,   79,  -91,   56,  -68,   80,  -92,  102,  114,  126,
       -138,  103, -115, -127, -139, -104,  116,  128,  140,    9,  -21,
         33,  -45,   10,   22,   34,   46,   11,   23,   35,  -47,  -57,
        -69,  -81,  -93,   58,   70,  -82,  -94,   59,  -71,  -83,   95,
       -105,  117, -129, -141,  106,  118,  130,  142, -107, -119, -131,
       -143),
         Tensor.fromData(Seq(4, 4, 3, 3),
            0,    1,    2,   48,   49,   50,   96,  -97,  -98,  -12,  -13,
        -14,  -60,   61,  -62,  108,  109, -110,   24,   25,  -26,  -72,
         73,  -74, -120, -121,  122,  -36,  -37,  -38,  -84,   85,   86,
       -132,  133, -134,    3,    4,   -5,   51,   52,  -53,   99, -100,
       -101,   15,   16,   17,   63,   64,  -65, -111, -112, -113,   27,
        -28,   29,  -75,   76,  -77,  123, -124,  125,   39,   40,   41,
         87,   88,  -89, -135, -136, -137,   -6,   -7,    8,  -54,  -55,
         56,  102,  103, -104,  -18,  -19,  -20,  -66,  -67,  -68,  114,
       -115,  116,   30,   31,   32,   78,   79,   80,  126, -127,  128,
        -42,   43,  -44,  -90,  -91,  -92, -138, -139,  140,    9,   10,
         11,  -57,   58,   59, -105,  106, -107,  -21,   22,   23,  -69,
         70,  -71,  117,  118, -119,   33,   34,   35,  -81,  -82,  -83,
       -129,  130, -131,  -45,   46,  -47,  -93,  -94,   95, -141,  142,
       -143),
         Tensor.fromData(Seq(4, 4, 3, 3),
            0,   48,   96,    1,   49,  -97,    2,   50,  -98,  -12,  -60,
        108,  -13,   61,  109,  -14,  -62, -110,   24,  -72, -120,   25,
         73, -121,  -26,  -74,  122,  -36,  -84, -132,  -37,   85,  133,
        -38,   86, -134,    3,   51,   99,    4,   52, -100,   -5,  -53,
       -101,   15,   63, -111,   16,   64, -112,   17,  -65, -113,   27,
        -75,  123,  -28,   76, -124,   29,  -77,  125,   39,   87, -135,
         40,   88, -136,   41,  -89, -137,   -6,  -54,  102,   -7,  -55,
        103,    8,   56, -104,  -18,  -66,  114,  -19,  -67, -115,  -20,
        -68,  116,   30,   78,  126,   31,   79, -127,   32,   80,  128,
        -42,  -90, -138,   43,  -91, -139,  -44,  -92,  140,    9,  -57,
       -105,   10,   58,  106,   11,   59, -107,  -21,  -69,  117,   22,
         70,  118,   23,  -71, -119,   33,  -81, -129,   34,  -82,  130,
         35,  -83, -131,  -45,  -93, -141,   46,  -94,  142,  -47,   95,
       -143),
          Tensor.fromData(Seq(4, 3, 3, 4),
            0,  -12,   24,  -36,   48,  -60,  -72,  -84,   96,  108, -120,
       -132,    1,  -13,   25,  -37,   49,   61,   73,   85,  -97,  109,
       -121,  133,    2,  -14,  -26,  -38,   50,  -62,  -74,   86,  -98,
       -110,  122, -134,    3,   15,   27,   39,   51,   63,  -75,   87,
         99, -111,  123, -135,    4,   16,  -28,   40,   52,   64,   76,
         88, -100, -112, -124, -136,   -5,   17,   29,   41,  -53,  -65,
        -77,  -89, -101, -113,  125, -137,   -6,  -18,   30,  -42,  -54,
        -66,   78,  -90,  102,  114,  126, -138,   -7,  -19,   31,   43,
        -55,  -67,   79,  -91,  103, -115, -127, -139,    8,  -20,   32,
        -44,   56,  -68,   80,  -92, -104,  116,  128,  140,    9,  -21,
         33,  -45,  -57,  -69,  -81,  -93, -105,  117, -129, -141,   10,
         22,   34,   46,   58,   70,  -82,  -94,  106,  118,  130,  142,
         11,   23,   35,  -47,   59,  -71,  -83,   95, -107, -119, -131,
       -143),
          Tensor.fromData(Seq(4, 3, 4, 3),
            0,   48,   96,  -12,  -60,  108,   24,  -72, -120,  -36,  -84,
       -132,    1,   49,  -97,  -13,   61,  109,   25,   73, -121,  -37,
         85,  133,    2,   50,  -98,  -14,  -62, -110,  -26,  -74,  122,
        -38,   86, -134,    3,   51,   99,   15,   63, -111,   27,  -75,
        123,   39,   87, -135,    4,   52, -100,   16,   64, -112,  -28,
         76, -124,   40,   88, -136,   -5,  -53, -101,   17,  -65, -113,
         29,  -77,  125,   41,  -89, -137,   -6,  -54,  102,  -18,  -66,
        114,   30,   78,  126,  -42,  -90, -138,   -7,  -55,  103,  -19,
        -67, -115,   31,   79, -127,   43,  -91, -139,    8,   56, -104,
        -20,  -68,  116,   32,   80,  128,  -44,  -92,  140,    9,  -57,
       -105,  -21,  -69,  117,   33,  -81, -129,  -45,  -93, -141,   10,
         58,  106,   22,   70,  118,   34,  -82,  130,   46,  -94,  142,
         11,   59, -107,   23,  -71, -119,   35,  -83, -131,  -47,   95,
       -143),
          Tensor.fromData(Seq(3, 3, 4, 4),
         0,    3,   -6,    9,  -12,   15,  -18,  -21,   24,   27,   30,
         33,  -36,   39,  -42,  -45,   48,   51,  -54,  -57,  -60,   63,
        -66,  -69,  -72,  -75,   78,  -81,  -84,   87,  -90,  -93,   96,
         99,  102, -105,  108, -111,  114,  117, -120,  123,  126, -129,
       -132, -135, -138, -141,    1,    4,   -7,   10,  -13,   16,  -19,
         22,   25,  -28,   31,   34,  -37,   40,   43,   46,   49,   52,
        -55,   58,   61,   64,  -67,   70,   73,   76,   79,  -82,   85,
         88,  -91,  -94,  -97, -100,  103,  106,  109, -112, -115,  118,
       -121, -124, -127,  130,  133, -136, -139,  142,    2,   -5,    8,
         11,  -14,   17,  -20,   23,  -26,   29,   32,   35,  -38,   41,
        -44,  -47,   50,  -53,   56,   59,  -62,  -65,  -68,  -71,  -74,
        -77,   80,  -83,   86,  -89,  -92,   95,  -98, -101, -104, -107,
       -110, -113,  116, -119,  122,  125,  128, -131, -134, -137,  140,
       -143),
         Tensor.fromData(Seq(3, 3, 4, 4),
        0,  -12,   24,  -36,    3,   15,   27,   39,   -6,  -18,   30,
        -42,    9,  -21,   33,  -45,   48,  -60,  -72,  -84,   51,   63,
        -75,   87,  -54,  -66,   78,  -90,  -57,  -69,  -81,  -93,   96,
        108, -120, -132,   99, -111,  123, -135,  102,  114,  126, -138,
       -105,  117, -129, -141,    1,  -13,   25,  -37,    4,   16,  -28,
         40,   -7,  -19,   31,   43,   10,   22,   34,   46,   49,   61,
         73,   85,   52,   64,   76,   88,  -55,  -67,   79,  -91,   58,
         70,  -82,  -94,  -97,  109, -121,  133, -100, -112, -124, -136,
        103, -115, -127, -139,  106,  118,  130,  142,    2,  -14,  -26,
        -38,   -5,   17,   29,   41,    8,  -20,   32,  -44,   11,   23,
         35,  -47,   50,  -62,  -74,   86,  -53,  -65,  -77,  -89,   56,
        -68,   80,  -92,   59,  -71,  -83,   95,  -98, -110,  122, -134,
       -101, -113,  125, -137, -104,  116,  128,  140, -107, -119, -131,
       -143),
          Tensor.fromData(Seq(3, 4, 3, 4), 0,    3,   -6,    9,   48,   51,  -54,  -57,   96,   99,  102,
       -105,  -12,   15,  -18,  -21,  -60,   63,  -66,  -69,  108, -111,
        114,  117,   24,   27,   30,   33,  -72,  -75,   78,  -81, -120,
        123,  126, -129,  -36,   39,  -42,  -45,  -84,   87,  -90,  -93,
       -132, -135, -138, -141,    1,    4,   -7,   10,   49,   52,  -55,
         58,  -97, -100,  103,  106,  -13,   16,  -19,   22,   61,   64,
        -67,   70,  109, -112, -115,  118,   25,  -28,   31,   34,   73,
         76,   79,  -82, -121, -124, -127,  130,  -37,   40,   43,   46,
         85,   88,  -91,  -94,  133, -136, -139,  142,    2,   -5,    8,
         11,   50,  -53,   56,   59,  -98, -101, -104, -107,  -14,   17,
        -20,   23,  -62,  -65,  -68,  -71, -110, -113,  116, -119,  -26,
         29,   32,   35,  -74,  -77,   80,  -83,  122,  125,  128, -131,
        -38,   41,  -44,  -47,   86,  -89,  -92,   95, -134, -137,  140,
       -143),
          Tensor.fromData(Seq(3, 4, 4, 3), 0,   48,   96,    3,   51,   99,   -6,  -54,  102,    9,  -57,
       -105,  -12,  -60,  108,   15,   63, -111,  -18,  -66,  114,  -21,
        -69,  117,   24,  -72, -120,   27,  -75,  123,   30,   78,  126,
         33,  -81, -129,  -36,  -84, -132,   39,   87, -135,  -42,  -90,
       -138,  -45,  -93, -141,    1,   49,  -97,    4,   52, -100,   -7,
        -55,  103,   10,   58,  106,  -13,   61,  109,   16,   64, -112,
        -19,  -67, -115,   22,   70,  118,   25,   73, -121,  -28,   76,
       -124,   31,   79, -127,   34,  -82,  130,  -37,   85,  133,   40,
         88, -136,   43,  -91, -139,   46,  -94,  142,    2,   50,  -98,
         -5,  -53, -101,    8,   56, -104,   11,   59, -107,  -14,  -62,
       -110,   17,  -65, -113,  -20,  -68,  116,   23,  -71, -119,  -26,
        -74,  122,   29,  -77,  125,   32,   80,  128,   35,  -83, -131,
        -38,   86, -134,   41,  -89, -137,  -44,  -92,  140,  -47,   95,
       -143),
            Tensor.fromData(Seq(3, 4, 3, 4), 0,  -12,   24,  -36,   48,  -60,  -72,  -84,   96,  108, -120,
       -132,    3,   15,   27,   39,   51,   63,  -75,   87,   99, -111,
        123, -135,   -6,  -18,   30,  -42,  -54,  -66,   78,  -90,  102,
        114,  126, -138,    9,  -21,   33,  -45,  -57,  -69,  -81,  -93,
       -105,  117, -129, -141,    1,  -13,   25,  -37,   49,   61,   73,
         85,  -97,  109, -121,  133,    4,   16,  -28,   40,   52,   64,
         76,   88, -100, -112, -124, -136,   -7,  -19,   31,   43,  -55,
        -67,   79,  -91,  103, -115, -127, -139,   10,   22,   34,   46,
         58,   70,  -82,  -94,  106,  118,  130,  142,    2,  -14,  -26,
        -38,   50,  -62,  -74,   86,  -98, -110,  122, -134,   -5,   17,
         29,   41,  -53,  -65,  -77,  -89, -101, -113,  125, -137,    8,
        -20,   32,  -44,   56,  -68,   80,  -92, -104,  116,  128,  140,
         11,   23,   35,  -47,   59,  -71,  -83,   95, -107, -119, -131,
       -143),
        Tensor.fromData(Seq(3, 4, 4, 3), 0,   48,   96,  -12,  -60,  108,   24,  -72, -120,  -36,  -84,
       -132,    3,   51,   99,   15,   63, -111,   27,  -75,  123,   39,
         87, -135,   -6,  -54,  102,  -18,  -66,  114,   30,   78,  126,
        -42,  -90, -138,    9,  -57, -105,  -21,  -69,  117,   33,  -81,
       -129,  -45,  -93, -141,    1,   49,  -97,  -13,   61,  109,   25,
         73, -121,  -37,   85,  133,    4,   52, -100,   16,   64, -112,
        -28,   76, -124,   40,   88, -136,   -7,  -55,  103,  -19,  -67,
       -115,   31,   79, -127,   43,  -91, -139,   10,   58,  106,   22,
         70,  118,   34,  -82,  130,   46,  -94,  142,    2,   50,  -98,
        -14,  -62, -110,  -26,  -74,  122,  -38,   86, -134,   -5,  -53,
       -101,   17,  -65, -113,   29,  -77,  125,   41,  -89, -137,    8,
         56, -104,  -20,  -68,  116,   32,   80,  128,  -44,  -92,  140,
         11,   59, -107,   23,  -71, -119,   35,  -83, -131,  -47,   95,
       -143)
        )
        val expectedGrad = Tensor.fromData(Seq(3,4,4,3),
           0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
       1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1,
       0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
       1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
       0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0)
        (resGrad zip expects) foreach { case ((res, grad), exp) =>
           Tensor.assertEqual(exp, res.toCPU())
           Tensor.assertEqual(expectedGrad, grad.toCPU())
        }
      }
    }
    runTest(permute)
  }*/

  testGPU("sumDim") {
    val sumDim = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-sum-dim"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2,3,2), -1,2,-3,4,-5,6, -7,8,-9,10,-11,12)
        val output = input.sum(1)
        val grad = gradR(x => x.sum(1).relu(false))(input)
        generate_comment("check")
        backend = BackendCPU()
        val expect = Tensor.fromData(Seq(2, 2), -9, 12, -27, 30)
        val expectGrad = Tensor.fromData(Seq(2, 3, 2), 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
        Tensor.assertEqual(expect, output.toCPU(), "expect and output are")
        Tensor.assertEqual(expectGrad, grad.toCPU(), "expect and output are")
      }
    }
    runTest(sumDim)
  }

  testGPU("relu") {
    val relu = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-relu"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        // TODO: Test NaN values.
        val input = Tensor.fromData(Seq(1,1,2,3), -1, 2, -3, 4, -5, 6)
        val result = input.relu()
        val grad = gradR(x => x.relu())(input)

        backend = BackendCPU()
        val expectedRes = Tensor.fromData(Seq(1,1,2,3), 0, 2, 0, 4, 0, 6)
        val expectedGrad = Tensor.fromData(Seq(1,1,2,3), 0, 1, 0, 1, 0, 1)
        Tensor.assertEqual(expectedRes, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(relu)
  }

  testGPU("tanh") {
    val tanh = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-tanh"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.randinit(Seq(1,1,2,3))
        val result = input.tanh()
        val grad = gradR(x => x.tanh())(input)
        val expected = Tensor.fill(Seq(1,1,2,3), 1) - result * result
        backend = BackendCPU()
        Tensor.assertEqual(expected.toCPU(), grad.toCPU())
      }
    }
    runTest(tanh)
  }

  testGPU("sigmoid") {
    val sigmoid = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-sigmoid"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.randinit(Seq(1,1,2,3))
        val result = input.sigmoid()
        val grad = gradR(x => x.sigmoid())(input)
        val expected = (Tensor.fill(Seq(1,1,2,3),1) - result) * result
        backend = BackendCPU()
        Tensor.assertEqual(expected.toCPU(), grad.toCPU())
      }
    }
    runTest(sigmoid)
  }

  testGPU("softmax") {
    val softmax = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-softmax"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.softmax_batch()
        val grad = gradR(x => x.softmax_batch())(input)
        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2, 3),
          0.0900305733f, 0.2447284758f, 0.6652409434f,
          0.0900305733f, 0.2447284758f, 0.6652409434f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.0000000107f, 0.0000000292f, 0.0000000793f,
          0.0000000107f, 0.0000000292f, 0.0000000793f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(softmax)
  }

  testGPU("log-softmax") {
    val logSoftmax = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-log-softmax"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.logSoftmaxB()
        val grad = gradR(x => x.logSoftmaxB())(input)
        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2, 3),
          -2.4076058865f, -1.4076058865f, -0.4076058865f,
          -2.4076061249f, -1.4076061249f, -0.4076061249f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.7299082279f, 0.2658145428f, -0.9957230091f,
          0.7299083471f, 0.2658147216f, -0.9957225323f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(logSoftmax)
  }

  testGPU("conv2D-forward") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1, 1, 4, 4), 1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8)
        val kernel = Tensor.fromData(Seq(1, 1, 2, 2), 1,2,3,4)
        val bias = Tensor.ones(1)
        val strides = Seq(2, 2)
        val pads = Seq(0,0,0,0)
        val (output, finputOption, _) = input.conv2D_batch(kernel, Some(bias), strides, pads)

        generate_comment("check")
        backend = BackendCPU()
        val expect = Tensor.fromData(Seq(1,1,2,2), 45, 65, 45, 65)
        Tensor.assertEqual(expect, output.toCPU(), "expect and output are")
      }
    }
    runTest(conv2D)
  }

  testGPU("conv2D-bias-grad") {
    val conv2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-conv2d-bias-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        generate_comment("input")
        val input = TensorR(Tensor.ones(1,1,4,4))
        val kernel = TensorR(Tensor.ones(1,1,3,3))
        val bias = TensorR(Tensor.zeros(1))
        val strides = Seq(1,1)
        val pads = Seq(0,0,0,0)
        def conv(x: TensorR) = {
          input.convBBP(kernel, Some(bias), strides, pads)
        }
        generate_comment("grad")
        gradR(conv)(Tensor.zeros(1))

        generate_comment("check")
        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,4,4),
          1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 4.0f, 4.0f, 2.0f, 2.0f, 4.0f, 4.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f)
        val expect_kernel_grad = Tensor.fill(Seq(1, 1, 3, 3), 4.0f)
        val expect_bias_grad = Tensor.fromData(Seq(1), 4.0f)

        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and input.gradient are")
        Tensor.assertEqual(expect_kernel_grad, kernel.d.toCPU(), "expect and kernel.gradient are")
        Tensor.assertEqual(expect_bias_grad, bias.d.toCPU(), "expect and bias.gradient are")
      }
    }
    runTest(conv2D)
  }

  testGPU("maxPool2D_batch") {
    val maxPool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-maxpool"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9)
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        val (output, _) = input.maxPool2D_batch(kernel, strides, None)

        backend = BackendCPU()
        val expect_output = Tensor.fromData(Seq(1,1,2,2), 5, 6, 8, 9)
        Tensor.assertEqual(expect_output, output.toCPU(), "expect and output are")
      }
    }
    runTest(maxPool2D)
  }

  testGPU("maxPool2D_batch_grad") {
    val maxPool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-maxpool-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9))
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        def lossFun(dummy: TensorR) = {
          input.maxPoolBK(kernel, strides, None)
        }
        gradR(lossFun)(Tensor.zeros(1))

        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,3,3), 0,0,0,0,1,1,0,1,1)
        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and output are")
      }
    }
    runTest(maxPool2D)
  }

  testGPU("averagePool2D_batch") {
    val averagePool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-averagePool"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9)
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        val output = input.averagePool2D_batch(kernel, strides, None)

        backend = BackendCPU()
        val expect_output = Tensor.fromData(Seq(1,1,2,2), 3, 4, 6, 7)
        Tensor.assertEqual(expect_output, output.toCPU(), "expect and output are")
      }
    }
    runTest(averagePool2D)
  }

  testGPU("averagePool2D_batch_grad") {
    val averagePool2D = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-averagePool-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9))
        val kernel = Seq(2,2)
        val strides = Seq(1,1)
        def lossFun(dummy: TensorR) = {
          input.averagePoolBK(kernel, strides, None)
        }
        gradR(lossFun)(Tensor.zeros(1))

        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,3,3), 0.25f,0.5f,0.25f,0.5f,1,0.5f,0.25f,0.5f,0.25f)
        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and output are")
      }
    }
    runTest(averagePool2D)
  }

  testGPU("dropout_batch") {
    val dropout = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-dropout"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(3,3,1,1),1,2,3,4,5,6,7,8,9)
        val prob = 0.0f
        val (output, _, _) = input.dropout(prob)

        backend = BackendCPU()
        Tensor.assertEqual(input.toCPU(), output.toCPU(), "expect and output are")
      }
    }
    runTest(dropout)
  }

  testGPU("dropout_batch_grad") {
    val dropout = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-dropout-grad"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = TensorR(Tensor.fromData(Seq(1,1,3,3),1,2,3,4,5,6,7,8,9))
        val prob = 0.0f
        def lossFun(dummy: TensorR) = {
          input.dropout(prob)
        }
        gradR(lossFun)(Tensor.zeros(1))

        backend = BackendCPU()
        val expect_input_grad = Tensor.fromData(Seq(1,1,3,3), 1,1,1,1,1,1,1,1,1)
        Tensor.assertEqual(expect_input_grad, input.d.toCPU(), "expect and output are")
      }
    }
    runTest(dropout)
  }

  testGPU("nll-loss") {
    val nllLoss = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-nll-loss"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(2, 3), 1, 2, 3, 4, 5, 6)
        val target: Rep[Array[Int]] = Array[Int](1, 0).toGPU(2)
        val result = input.logSoftmaxB().nllLossB(target)
        val grad = gradR(x => x.logSoftmaxB().nllLossB(target))(input)

        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2), 1.4076058865f, 2.4076061249f)
        val expectedGrad = Tensor.fromData(Seq(2, 3),
          0.0900305808f, -0.7552714944f, 0.6652410030f,
          -0.9099694490f, 0.2447284311f, 0.6652408242f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(nllLoss)
  }

  testGPU("repeat0") {
    val repeat0 = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-repeat0"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(5,2,2), 1,2,3,4, 1,2,3,4, 1,2,3,4, 0,0,0,0, 0,0,0,0)
        val result = input.repeat0(2)
        val grad = gradR(x => x.repeat0(2))(input)

        backend = BackendCPU()
        generate_comment("check for correctness")
        val expectedResult = Tensor.fromData(Seq(3,3,2,2),
          1,2,3,4, 1,2,3,4, 1,2,3,4,
          1,2,3,4, 1,2,3,4, 0,0,0,0,
          1,2,3,4, 0,0,0,0, 0,0,0,0)
        val expectedGrad = Tensor.fromData(Seq(5,2,2), 1,1,1,1, 2,2,2,2, 3,3,3,3, 0,0,0,0, 0,0,0,0)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(repeat0)
  }

  testGPU("concat_grad") {
    val concat = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-concat-grad"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input1 = Tensor.fromData(Seq(2,3,2,2), 1,2,3,4,5,6,7,8,1,2,3,4, 5,6,7,8,1,2,3,4,5,6,7,8)
        val input2 = Tensor.fromData(Seq(2,4,2,2), 3,4,5,6,3,4,5,6,3,4,5,6,3,4,5,6, 6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9)
        val result = input1.concat(1, input2)
        val input1R = TensorR(input1)
        val input2R = TensorR(input2)
        val grad = gradR(x => input1R.concat(1, input2R))(Tensor.zeros(1))

        backend = BackendCPU()
        val expectedResult = Tensor.fromData(Seq(2,7,2,2), 1,2,3,4,5,6,7,8,1,2,3,4,3,4,5,6,3,4,5,6,3,4,5,6,3,4,5,6,
          5,6,7,8,1,2,3,4,5,6,7,8,6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9)
        val expectedGrad1 = Tensor.fill(Seq(2,3,2,2), 1.0f)
        val expectedGrad2 = Tensor.fill(Seq(2,4,2,2), 1.0f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad1, input1R.d.toCPU())
        Tensor.assertEqual(expectedGrad2, input2R.d.toCPU())
      }
    }
    runTest(concat)
  }

  testGPU("concat_big") {
    val concat = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-concat-grad"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input1 = Tensor.fill(Seq(100,3,32,32), 1.0f)
        val input2 = Tensor.fill(Seq(100,3,32,32), 1.0f)
        val result = input1.concat(1, input2)
        val input1R = TensorR(input1)
        val input2R = TensorR(input2)
        val grad = gradR(x => input1R.concat(1, input2R))(Tensor.zeros(1))

        backend = BackendCPU()
        val expectedResult = Tensor.fill(Seq(100,6,32,32), 1.0f)
        val expectedGrad1 = Tensor.fill(Seq(100, 3, 32, 32), 1.0f)
        val expectedGrad2 = Tensor.fill(Seq(100, 3, 32, 32), 1.0f)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad1, input1R.d.toCPU())
        Tensor.assertEqual(expectedGrad2, input2R.d.toCPU())
      }
    }
    runTest(concat)
  }

  testGPU("sum") {
    val sum = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-sum"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.fromData(Seq(1, 1, 2, 3), 1, 2, 3, 4, 5, 6)
        val result = input.sum()
        val grad = gradR(x => x.sum())(input)

        backend = BackendCPU()
        val expectedResult = Tensor.scalar(21)
        val expectedGrad = Tensor.fromData(Seq(1, 1, 2, 3), 1, 1, 1, 1, 1, 1)
        Tensor.assertEqual(expectedResult, result.toCPU())
        Tensor.assertEqual(expectedGrad, grad.toCPU())
      }
    }
    runTest(sum)
  }

  testGPU("batch-norm-inference") {
    val batchNorm = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-batch-norm"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val input = Tensor.ones(3,2,3,3)
        val scale = Tensor.fromData(Seq(1, 2, 1, 1), 3.0f, 3.0f)
        val bias = Tensor.fromData(Seq(1, 2, 1, 1), 2.0f, 2.0f)
        val runningMean = Tensor.fromData(Seq(1, 2, 1, 1), 3.0f, 3.0f)
        val runningVar = Tensor.fromData(Seq(1, 2, 1, 1), 4.0f, 4.0f)
        val result = input.batchNormInference(scale, bias, runningMean, runningVar)
        backend = BackendCPU()
        val expectedResult = Tensor.fill(Seq(3,2,3,3), -1.0f)
        Tensor.assertEqual(expectedResult, result.toCPU(), tal = 0.0001f)
      }
    }
    runTest(batchNorm)
  }

  testGPU("batch-norm-training") {
    val batchNorm = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-batch-norm-training"

      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        Tensor.randseed(42)
        val input = Tensor.rand(3, 2, 3, 3)
        val scale = TensorR(Tensor.fromData(Seq(1, 2, 1, 1), 2, 2))
        val bias = TensorR(Tensor.fromData(Seq(1, 2, 1, 1), -1, -1))
        val runningMean = Tensor.ones(1,2,1,1)
        val runningVar = Tensor.ones(1, 2, 1, 1)
        val result = Tensor.zeros(3, 2, 3, 3)
        def lossFun(x: TensorR) = {
          val y = x.batchNorm(scale, bias, runningMean, runningVar)
          result.copy_data(y.x)
          y.relu()
        }
        val grad = gradR(lossFun)(input)

        backend = BackendCPU()
        // result.toCPU().print("result")
        // grad.toCPU().print("grad")
        // TODO (Fei Wang) need to be confirmed by PyTorch
        val expectedResult = Tensor.fromData(Seq(3, 2, 3, 3),
          -4.70055f, -2.42800f, 0.33646f,
          -1.71884f, -3.37612f, -3.03992f,
          -0.07803f, 1.66236f, -2.64498f,
          -4.40594f, -2.17456f, 0.45260f,
          -2.48382f, -3.67889f, -3.86906f,
          0.42170f, -4.03358f, -0.95436f,
          -0.62918f, 0.73989f, 2.56785f,
          -3.27966f, -1.47426f, -0.98174f,
          -0.55874f, -0.70138f, 0.62703f,
          -0.44310f, 0.65435f, -0.73941f,
          -2.76659f, 0.87396f, 1.42567f,
          1.76504f, -2.91541f, -3.78245f,
          -3.30510f, 1.85414f, 2.58171f,
          -0.99299f, 2.04517f, -2.28546f,
          -2.79128f, -3.18326f, -1.24514f,
          -2.01499f, 1.94021f, -0.85590f,
          1.60013f, -0.91634f, -2.54044f,
          1.48045f, 0.51965f, 0.44109f)
        val expectedGrad = Tensor.fromData(Seq(3, 2, 3, 3),
          3.09040f, -0.20212f, 3.45742f,
          -1.22957f, 1.17154f, 0.68444f,
          -3.60681f, 1.53643f, 0.11224f,
          1.99813f, -1.06226f, 1.89603f,
          -0.63810f, 1.00096f, 1.26179f,
          1.93841f, 1.48743f, -2.73582f,
          -2.80830f, 2.87292f, 0.22452f,
          1.03179f, -1.58392f, -2.29750f,
          -2.91035f, -2.70368f, 3.03643f,
          -3.43702f, 1.61933f, -3.03063f,
          -0.25029f, 1.31812f, 0.56144f,
          0.09597f, -0.04617f, 1.14300f,
          1.06865f, 1.25857f, 0.20445f,
          -2.28120f, 0.98180f, -0.40863f,
          0.32422f, 0.89211f, -1.91587f,
          -1.28113f, -0.14428f, -2.87085f,
          0.32216f, -2.78796f, -0.56045f,
          0.48630f, 1.80406f, 1.91181f)
        Tensor.assertEqual(expectedResult, result.toCPU(), tal = 0.0001f)
        Tensor.assertEqual(expectedGrad, grad.toCPU(), tal = 0.0001f)
      }
    }
    runTest(batchNorm)
  }

  /*
  testGPU("rnn-forward") {
    val rnnInference = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-rnn-forward"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val inputSize = 10
        val hiddenSize = 40
        val numLayers = 2
        val seqLength = 5
        val batchSize = 3
        val bidirectional = false
        val numDirections = if (bidirectional) 2 else 1

        def getParameterSize(): Int = {
          val gateSize = hiddenSize
          val w_ih_size = gateSize * inputSize + (numLayers - 1) * gateSize * hiddenSize * numDirections
          val w_hh_size = numLayers * gateSize * hiddenSize
          val b_ih_size = numLayers * gateSize
          val b_hh_size = numLayers * gateSize
          w_ih_size + w_hh_size + b_ih_size + b_hh_size
        }

        val x = Tensor.ones(seqLength, batchSize, inputSize)
        val hx = Tensor.ones(numLayers, batchSize, hiddenSize)
        val w = Tensor.fill(Seq(getParameterSize), 0.01f)
        val res1 = BackendCudnn().cudnnRNNForwardInference(RnnReluMode, x, Some(hx), cx = None, w, numLayers, hiddenSize)
        val res2 = BackendCudnn().cudnnRNNForwardTraining(RnnReluMode, x, Some(hx), cx = None, w, numLayers, hiddenSize)._1
        backend = BackendCPU()
        res2.toCPU().print()
        Tensor.assertEqual(res1.toCPU(), res2.toCPU())
      }
    }
    runTest(rnnInference)
  }
  */

  testGPU("rnn-module") {
    val rnnModule = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-rnn-module"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        Tensor.randseed(42)
        val inputSize = 10
        val hiddenSize = 40
        val numLayers = 2
        val seqLength = 5
        val batchSize = 3
        val bidirectional = true
        val numDirections = if (bidirectional) 2 else 1

        val input = Tensor.ones(seqLength, batchSize, inputSize)
        val h0 = TensorR(Tensor.ones(numLayers * numDirections, batchSize, hiddenSize))
        val rnn = RNNRelu(inputSize, hiddenSize, numLayers, bidirectional = bidirectional)

        /*
        def lossFun(input: TensorR) = {
          val res = rnn(input)
          res.sum()
        }
        val loss = gradR_loss(lossFun)(input)
        */

        def lossFun(input: TensorR) = {
          rnn(input, Some(h0))
        }
        val dInput = gradR(lossFun)(input)

        backend = BackendCPU()
        dInput.toCPU().print()

        for (layer <- (0 until numLayers): Range) {
          printf("w_ih[%d]\\n", layer)
          rnn.w_ih(layer).d.toCPU().printHead()
          printf("w_hh[%d]\\n", layer)
          rnn.w_hh(layer).d.toCPU().printHead()
          printf("b_ih[%d]\\n", layer)
          rnn.b_ih(layer).d.toCPU().printHead()
          printf("b_hh[%d]\\n", layer)
          rnn.b_hh(layer).d.toCPU().printHead()

          printf("w_ih_reverse[%d]\\n", layer)
          rnn.w_ih_reverse(layer).d.toCPU().printHead()
          printf("w_hh_reverse[%d]\\n", layer)
          rnn.w_hh_reverse(layer).d.toCPU().printHead()
          printf("b_ih_reverse[%d]\\n", layer)
          rnn.b_ih_reverse(layer).d.toCPU().printHead()
          printf("b_hh_reverse[%d]\\n", layer)
          rnn.b_hh_reverse(layer).d.toCPU().printHead()
        }
      }
    }
    runTest(rnnModule)
  }

  testGPU("lstm-module") {
    val lstmModule = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-lstm-module"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        Tensor.randseed(42)
        val inputSize = 2
        val hiddenSize = 2
        val numLayers = 2
        val seqLength = 2
        val batchSize = 2
        val bidirectional = true
        val numDirections = if (bidirectional) 2 else 1

        val input = Tensor.ones(seqLength, batchSize, inputSize)
        val h0 = TensorR(Tensor.ones(numLayers * numDirections, batchSize, hiddenSize))
        val c0 = TensorR(Tensor.ones(numLayers * numDirections, batchSize, hiddenSize))
        val rnn = LSTM(inputSize, hiddenSize, numLayers, bidirectional = bidirectional)

        // Test parameter registration.
        rnn.registerParameters("lstm")
        val expectedParameterCount = numLayers * numDirections * 4
        assert(rnn.parameters.size == expectedParameterCount)

        def lossFun(input: TensorR) = {
          rnn(input, Some(h0), Some(c0))
          rnn(input, Some(h0), Some(c0))
        }
        val dInput = gradR(lossFun)(input)

        backend = BackendCPU()
        dInput.toCPU().print()

        for (layer <- (0 until numLayers): Range) {
          printf("w_ih[%d].grad\\n", layer)
          rnn.w_ih(layer).d.toCPU().print()
          printf("w_hh[%d].grad\\n", layer)
          rnn.w_hh(layer).d.toCPU().print()
          printf("b_ih[%d].grad\\n", layer)
          rnn.b_ih(layer).d.toCPU().print()
          printf("b_hh[%d].grad\\n", layer)
          rnn.b_hh(layer).d.toCPU().print()

          printf("w_ih_reverse[%d].grad\\n", layer)
          rnn.w_ih_reverse(layer).d.toCPU().print()
          printf("w_hh_reverse[%d].grad\\n", layer)
          rnn.w_hh_reverse(layer).d.toCPU().print()
          printf("b_ih_reverse[%d].grad\\n", layer)
          rnn.b_ih_reverse(layer).d.toCPU().print()
          printf("b_hh_reverse[%d].grad\\n", layer)
          rnn.b_hh_reverse(layer).d.toCPU().print()
        }
      }
    }
    runTest(lstmModule)
  }

  testGPU("lstm-opt") {
    val lstmModule = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-lstm-opt"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        Tensor.randseed(42)
        val inputSize = 2
        val hiddenSize = 2
        val numLayers = 2
        val seqLength = 2
        val batchSize = 2
        val bidirectional = true
        val numDirections = if (bidirectional) 2 else 1

        val input = Tensor.ones(seqLength, batchSize, inputSize)
        val rnn = LSTM(inputSize, hiddenSize, numLayers, bidirectional = bidirectional)
        val opt = SGD(rnn, learning_rate = 0.1f)

        // Test parameter registration.
        // System.out.println(rnn.parameters)
        // System.out.println(rnn.parameters.size)
        val expectedParameterCount = numLayers * numDirections * 4
        assert(rnn.parameters.size == expectedParameterCount)

        def lossFun(input: TensorR) = {
          rnn(input)
        }
        val dInput = gradR(lossFun)(input)

        // Print w_ih parameters before SGD step.
        for (layer <- (0 until numLayers): Range) {
          printf("w_ih[%d]\\n", layer)
          rnn.w_ih(layer).x.toCPU().print()
        }

        opt.step()

        // Print w_ih parameters after SGD step.
        for (layer <- (0 until numLayers): Range) {
          printf("w_ih[%d]\\n", layer)
          rnn.w_ih(layer).x.toCPU().print()
        }
      }
    }
    runTest(lstmModule)
  }

  testGPU("ctc-loss") {
    val ctcLoss = new LanternDriverCudnn[String, Unit] {
      override val fileName = "lantern-cudnn-ctc-loss"
      @virtualize
      def snippet(a: Rep[String]): Rep[Unit] = {
        val inputLength = 50
        val batchSize = 16
        val alphabetSize = 20

        // val logProbs = Tensor.ones(inputLength, batchSize, alphabetSize)
        // Note: `probs` should be the result of `logProbs.softmax(dim = 2)`.
        val probs = TensorR(Tensor.fill(Seq(inputLength, batchSize, alphabetSize), 0.05f))
        val target = Array(Seq.fill(batchSize * 2)(5).map(unit(_)): _*)
        val inputLengths = Array(Seq.fill(batchSize)(inputLength - 21).map(unit(_)): _*)
        inputLengths(3) = inputLength
        val targetLengths = Array(Seq.fill(batchSize)(2).map(unit(_)): _*)
        val loss = probs.ctcLoss(inputLengths, target, targetLengths)

        backend = BackendCPU()
        loss.toCPU().print()
      }
    }
    runTest(ctcLoss)
  }
}
