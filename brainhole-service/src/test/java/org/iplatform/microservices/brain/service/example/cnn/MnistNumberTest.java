/*
 * 采用更复杂的Lenet算法来处理MNIST数据集，可以达到99%的准确率。Lenet是一种深度卷积网络。
 * */
package org.iplatform.microservices.brain.service.example.cnn;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.iplatform.microservices.brain.service.example.utilities.DataUtilities;
import org.iplatform.microservices.brain.service.multiLayernetwork.listeners.LossScoreIterationListener;
import org.iplatform.microservices.brain.service.service.NeuralNetworksMonitorService;
import org.iplatform.microservices.brain.util.INDArrayUtil;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.lang.reflect.Array;
import java.util.*;

/*
* https://raw.githubusercontent.com/myleott/mnist_png/master/mnist_png.tar.gz
* */
public class MnistNumberTest {
    private static final Logger log = LoggerFactory.getLogger(MnistNumberTest.class);
    private String tempDir = System.getProperty("java.io.tmpdir");
    private String DATA_PATH = FilenameUtils.concat(tempDir, "brainhole/mnist_number/");
    private String MODEL_PATH = FilenameUtils.concat(tempDir, "brainhole/mnist_number/mnist_number.model");

    @Test
    public void testModelFit() throws Exception {

        NeuralNetworksMonitorService monitor = new NeuralNetworksMonitorService();
        monitor.init();

        //解压数据
        File dir = new File(DATA_PATH);
        if(!dir.exists()){
            dir.mkdirs();
            log.info(String.format("解压数据到 %s",DATA_PATH));
            DataUtilities.extractTarGz("data/mnist_number/mnist_png.tar.gz",DATA_PATH);
        }else{
            log.info(String.format("数据 %s 已经存在",DATA_PATH   ));
        }

        //加载学习训练数据
        int height = 28;
        int width = 28;
        int channels = 1;
        int seed = 123;
        int batchSize = 128;
        int outputNum = 10;
        int numEpochs = 15;
        Random randNumGen = new Random(seed);

        File trainData = new File(DATA_PATH + "/mnist_png/training");
        File testData = new File(DATA_PATH + "/mnist_png/testing");

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        // Initialize the record reader
        // add a listener, to extract the name
        //recordReader.setListeners(new LogRecordListener());
        ImageRecordReader tranRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        tranRecordReader.initialize(train);
        DataSetIterator tranDataIter = new RecordReaderDataSetIterator(tranRecordReader,batchSize,1,outputNum);
        // Scale pixel values to 0-1
        DataNormalization tranScaler = new ImagePreProcessingScaler(0,1);
        tranScaler.fit(tranDataIter);
        tranDataIter.setPreProcessor(tranScaler);

        ImageRecordReader testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        testRecordReader.initialize(test);
        DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader,batchSize,1,outputNum);
        DataNormalization testScaler = new ImagePreProcessingScaler(0,1);
        testScaler.fit(testDataIter);
        testDataIter.setPreProcessor(testScaler);

//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(rngseed)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .iterations(1)
//                .learningRate(0.006)
//                .updater(new Nesterovs(0.9))
//                .regularization(true).l2(1e-4)
//                .list()
//                .layer(0, new DenseLayer.Builder()
//                        .nIn(height * width)
//                        .nOut(100)
//                        .activation(Activation.RELU)
//                        .weightInit(WeightInit.XAVIER)
//                        .build())
//                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nIn(100)
//                        .nOut(outputNum)
//                        .activation(Activation.SOFTMAX)
//                        .weightInit(WeightInit.XAVIER)
//                        .build())
//                .pretrain(false).backprop(true)
//                .setInputType(InputType.convolutional(height,width,channels))
//                .build();

        int nChannels = 1; // Number of input channels
        int iterations = 1; // Number of training iterations
        Map<Integer, Double> lrSchedule = new HashMap<>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations) // Training iterations as above
                .regularization(true).l2(0.0005)
                .learningRate(.01)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule).learningRateSchedule(lrSchedule)
                .weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0,
                        new ConvolutionLayer.Builder(5, 5)
                                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                .layer(1,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
                                .build())
                .layer(2,
                        new ConvolutionLayer.Builder(5, 5)
                                //Note that nIn need not be specified in later layers
                                .stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                .layer(3,
                        new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
                                .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
                .layer(5,
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
                                .activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(monitor.getStatsListener(),new LossScoreIterationListener(10));

        long s = System.currentTimeMillis();
        Evaluation eval =  null;
        for(int i = 0; i<numEpochs; i++){
            model.fit(tranDataIter);
            eval = model.evaluate(testDataIter);
            testDataIter.reset();
        }
        long e = System.currentTimeMillis();
        log.info("+-----------------------------------------------------+");
        log.info("| 准确率(Accuracy):" + eval.accuracy());// 准确识别出来的数量/数据总数量
        log.info("| 精确率(Precision):" + eval.precision());// 准确正例的数量／(真正例数量+假正例数量)
        log.info("| 召回率(Recall):" + eval.recall());//真正例的数量/(真正例数量+假负例数量)
        log.info("| 评价分(F1):" + eval.f1());//精确率和召回率的加权平均值
        log.info("+-----------------------------------------------------+");
        log.info("+ 耗时:" + (e - s) / 1000 + "秒");
        log.info("+-----------------------------------------------------+");

        ModelSerializer.writeModel(model,MODEL_PATH,false);
        log.info(String.format("存储模型到 %s",MODEL_PATH));
        monitor.waitTermination(1000*3600);
    }

    @Test
    public void testPredict() throws Exception {

        int height = 28;
        int width = 28;
        int channels = 1;

        // recordReader.getLabels()
        // In this version Labels are always in order
        // So this is no longer needed
        //List<Integer> labelList = Arrays.asList(2,3,7,1,6,4,0,5,8,9);
        List<Integer> labelList = Arrays.asList(0,1,2,3,4,5,6,7,8,9);


        //加载神经网络模型
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(MODEL_PATH);

        long s = System.currentTimeMillis();
        File dataDir = new File(DATA_PATH);
        List<String> files = getDirectory(dataDir,null);
        log.info("+-----------------------------------------------------+");
        log.info("图片 \t 识别数字");
        log.info("+-----------------------------------------------------+");
        int succeed=0;
        int failed=0;
        for(String filepath : files){
            //要预测的文件
            File file = new File(filepath);
            // Use NativeImageLoader to convert to numerical matrix
            NativeImageLoader loader = new NativeImageLoader(height, width, channels);
            INDArray image = loader.asMatrix(file);
            DataNormalization scaler = new ImagePreProcessingScaler(0,1);
            scaler.transform(image);

            INDArray output = model.output(image);
            int maxScoreIndex = INDArrayUtil.getMaxIndexFloatArrayFromSlice(output);
            double confidence = output.getDouble(maxScoreIndex);
            int predictValue = labelList.get(maxScoreIndex);
            String shortfilepath = filepath.replace(DATA_PATH,"");
            shortfilepath = shortfilepath.replace("mnist_png/training/","");
            shortfilepath = shortfilepath.replace("mnist_png/testing/","");
            int realValue = Integer.valueOf(shortfilepath.substring(0,1));
            if(realValue!=predictValue){
                log.info(String.format("错误 %s \t %d \t %.2f",filepath,predictValue,confidence));
                failed++;
            }else{
                log.info(String.format("正确 %s \t %d \t %.2f",filepath,predictValue,confidence));
                succeed++;
            }
        }
        long e = System.currentTimeMillis();
        log.info("+-----------------------------------------------------+");
        log.info(String.format("图片总数 %d",files.size()));
        log.info(String.format("正确识别 %d",succeed));
        log.info(String.format("错误识别 %d",failed));
        log.info(String.format("平均耗时 %d ms",(e-s)/files.size()));
        log.info(String.format("总耗时 %d s",(e-s)/1000));
        log.info(String.format("识别率 %.2f%%",((float)succeed/(succeed+failed))*100));
        log.info("+-----------------------------------------------------+");
    }

    private List<String> getDirectory(File file,List<String> files) {
        if(files==null){
            files = new ArrayList<String>();
        }
        File flist[] = file.listFiles();
        if (flist == null || flist.length == 0) {
            return files;
        }
        for (File f : flist) {
            if (f.isDirectory()) {
                files = getDirectory(f,files);
            } else {
                if(f.getAbsolutePath().endsWith(".png")){
                    files.add(f.getAbsolutePath());
                }
            }
        }
        return files;
    }

    @Test
    public void t(){
        log.info(String.format("识别率 %.2f %%",((float)68042/70000)*100f));
    }
}