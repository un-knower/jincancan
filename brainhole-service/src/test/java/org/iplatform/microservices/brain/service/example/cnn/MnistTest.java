/*
 * 采用更复杂的Lenet算法来处理MNIST数据集，可以达到99%的准确率。Lenet是一种深度卷积网络。
 * */
package org.iplatform.microservices.brain.service.example.cnn;

import java.util.HashMap;
import java.util.Map;

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
import org.iplatform.microservices.brain.service.multiLayernetwork.listeners.LossScoreIterationListener;
import org.iplatform.microservices.brain.service.service.NeuralNetworksMonitorService;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MnistTest {
    private static final Logger log = LoggerFactory.getLogger(MnistTest.class);


    public static void main(String[] args) throws Exception {

        NeuralNetworksMonitorService monitor = new NeuralNetworksMonitorService();
        monitor.init();

        int nChannels = 1; // Number of input channels
        int outputNum = 10; // The number of possible outcomes
        int batchSize = 64; // Test batch size
        int nEpochs = 1; // Number of training epochs
        int iterations = 1; // Number of training iterations
        int seed = 123; //

        log.info("加载数据");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

        log.info("创建网络模型");
        //定义学习速率计划
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

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)
        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(monitor.getStatsListener(),new LossScoreIterationListener(1));
        long s = System.currentTimeMillis();
        Evaluation eval =  null;
        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);
            eval = model.evaluate(mnistTest);
            mnistTest.reset();
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

        monitor.waitTermination(1000*3600);
    }
}