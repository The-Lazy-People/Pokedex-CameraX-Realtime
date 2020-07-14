package com.abhitom.camerax

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.ViewTreeObserver
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.bottomsheet_activity.*
import kotlinx.android.synthetic.main.content_main_activity.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min


class MainActivity : AppCompatActivity() {
    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private val TAG = "CameraXBasic"

    private lateinit var cameraExecutor: ExecutorService

    protected var tflite: Interpreter? = null
    private var inputImageBuffer: TensorImage? = null
    private var imageSizeX = 0
    private var imageSizeY = 0
    private var outputProbabilityBuffer: TensorBuffer? = null
    private var probabilityProcessor: TensorProcessor? = null
    private val IMAGE_MEAN = 0.0f
    private val IMAGE_STD = 255.0f
    private val PROBABILITY_MEAN = 0.0f
    private val PROBABILITY_STD = 255.0f
    private var labels: List<String>? = null


    @RequiresApi(Build.VERSION_CODES.M)
    @androidx.camera.core.ExperimentalGetImage
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if (ActivityCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(Manifest.permission.CAMERA),1001)
        }
        else{
            startCamera()
            cameraExecutor = Executors.newSingleThreadExecutor()
        }
        try {
            tflite = Interpreter(this.loadmodelfile(this)!!)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun Image.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer // Y
        val uBuffer = planes[1].buffer // U
        val vBuffer = planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
    @androidx.camera.core.ExperimentalGetImage
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            preview = Preview.Builder()
                .build()
            imageCapture = ImageCapture.Builder()
                .build()
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()


            viewFinder.afterMeasured {
                viewFinder.setOnTouchListener { _, event ->
                    return@setOnTouchListener when (event.action) {
                        MotionEvent.ACTION_DOWN -> {
                            true
                        }
                        MotionEvent.ACTION_UP -> {
                            val factory: MeteringPointFactory = SurfaceOrientedMeteringPointFactory(
                                viewFinder.width.toFloat(), viewFinder.height.toFloat()
                            )
                            val autoFocusPoint = factory.createPoint(event.x, event.y)
                            try {
                                camera?.cameraControl?.startFocusAndMetering(
                                    FocusMeteringAction.Builder(
                                        autoFocusPoint,
                                        FocusMeteringAction.FLAG_AF
                                    ).apply {
                                        //focus only when the user tap the preview
                                        disableAutoCancel()
                                    }.build()
                                )
                            } catch (e: CameraInfoUnavailableException) {
                                Log.d("ERROR", "cannot access camera", e)
                            }
                            true
                        }
                        else -> false // Unhandled event.
                    }
                }
            }





            imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), ImageAnalysis.Analyzer { image ->
                val matrix = Matrix()
                matrix.postRotate(90f)
                val scaledBitmap = Bitmap.createScaledBitmap(image.image!!.toBitmap(), image.image!!.toBitmap().width, image.image!!.toBitmap().height, true)
                val rotatedBitmap = Bitmap.createBitmap(
                    scaledBitmap,
                    0,
                    0,
                    scaledBitmap.width,
                    scaledBitmap.height,
                    matrix,
                    true
                )
                classify( rotatedBitmap)
                image.close()

            })
            val cameraSelector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture,imageAnalysis)
                preview?.setSurfaceProvider(viewFinder.createSurfaceProvider(camera?.cameraInfo))
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))

    }
    inline fun View.afterMeasured(crossinline block: () -> Unit) {
        viewTreeObserver.addOnGlobalLayoutListener(object : ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                if (measuredWidth > 0 && measuredHeight > 0) {
                    viewTreeObserver.removeOnGlobalLayoutListener(this)
                    block()
                }
            }
        })
    }
    @Throws(IOException::class)
    private fun loadmodelfile(activity: Activity): MappedByteBuffer? {
        val fileDescriptor =
            activity.assets.openFd("converted_model.tflite")
        val inputStream =
            FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startoffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startoffset,
            declaredLength
        )
    }

    private fun loadImage(bitmap: Bitmap): TensorImage? {
        inputImageBuffer!!.load(bitmap)
        val cropSize = min(bitmap.width, bitmap.height)
        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(getPreprocessNormalizeOp())
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    private fun getPreprocessNormalizeOp(): TensorOperator? {
        return NormalizeOp(IMAGE_MEAN, IMAGE_STD)
    }

    private fun getPostprocessNormalizeOp(): TensorOperator? {
        return NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)
    }

    private fun showresult() {
        try {
            labels = FileUtil.loadLabels(this, "output.txt")
        } catch (e: Exception) {
            e.printStackTrace()
        }
        val labeledProbability: Map<String, Float> =
            TensorLabel(labels!!.toList(), probabilityProcessor!!.process(outputProbabilityBuffer))
                .mapWithFloatValue
        var v1=0f
        var s1=""
        var v2=0f
        var s2=""
        var v3=0f
        var s3=""
        for ((key, value) in labeledProbability) {
            if(value>v1){
                v1=value
                s1=key
            }
            if(value>v2 && value<v1){
                v2=value
                s2=key
            }
            if(value>v3 && value<v2){
                v3=value
                s3=key
            }
        }
        tvPokemonName1.text=s1
        tvPokemonName2.text=s2
        tvPokemonName3.text=s3
    }

    private fun classify(bitmap: Bitmap)
    {
        if (tflite==null)return
        val imageTensorIndex = 0
        val imageShape: IntArray? =
            tflite?.getInputTensor(imageTensorIndex)?.shape() // {1, height, width, 3}
        imageSizeY = imageShape!![1]
        imageSizeX = imageShape[2]
        val imageDataType: DataType = tflite!!.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape: IntArray =
            tflite!!.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}
        val probabilityDataType: DataType =
            tflite!!.getOutputTensor(probabilityTensorIndex).dataType()
        inputImageBuffer = TensorImage(imageDataType)
        outputProbabilityBuffer =
            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
        probabilityProcessor = TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build()
        inputImageBuffer = loadImage(bitmap)
        tflite!!.run(
            inputImageBuffer!!.buffer,
            outputProbabilityBuffer!!.buffer.rewind()
        )
        showresult()
    }
}