package org.iplatform.microservices.brain.service.example.rnn.seqclassification;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import au.com.bytecode.opencsv.CSVReader;

/**
 * 使用LSTM 循环神经网络（RNN）对序列数据数据进行分类
 *
 * 这个例子通过单变量时间序列数据学习6个分类.
 * 分类: 0正常, 1循环, 2趋势增加, 3趋势下降, 4向上转移, 5向下转移
 * 
 * 数据描述：前60列是时间序列数据，61列是分类
 * 实例代码处理流程如下:
 * 1. 下载数据集合
 *    (a) 分割数据集600条进行拆分，训练数据450条，测试数据150条
 *        说明：原始数据每个时间序列一行数据，数据通过制表符分割
 *    (b) 格式化数据，生成适合CSVSequenceRecordReader加载的数据格式
 *        说明: 每个时间序列生成一个csv文件和这个文件所对应的标签
 *        例如, train/features/0.csv 是特征文件，train/labels/0.csv 是这个特征文件的分类标签
 *        因为数据是一个单变量的时间序列，所以我们每个csv文件只有一列, 每列包含多个行，每行代表一个时间步长
 *        因为每个时间序列只有一个标签分类，所以label下的csv文件只有一个值
 *
 * 2. 使用CSVSequenceRecordReader，SequenceRecordReaderDataSetIterator加载训练数据集
 *    用法参见: http://deeplearning4j.org/usingrnns#data
 *
 * 3. 规范化数据，使用NormalizerStandardize规范化数据
 *
 * 4. 配置神经网络
 *    因为数据集比较小，所以我们不能使用大型网络和过多的参数，所以我们使用一个小型LSTM和RNN作为输出层
 *
 * 5. 对网络进行40次训练
 *    每次训练都会评估模型准确性和F1值，并打印出来
 */
public class UCISequenceClassificationExample {
    private static final Logger log = LoggerFactory.getLogger(UCISequenceClassificationExample.class);
    //训练数据临时存储路径

    private static File baseDir;
    private static File baseTrainDir;
    private static File featuresDirTrain;
    private static File labelsDirTrain;
    private static File baseTestDir;
    private static File featuresDirTest;
    private static File labelsDirTest;

    @Test
    public void test() throws Exception {
        downloadUCIData();

        // ----- Load the training data -----
        //Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        trainData.setPreProcessor(normalizer);


        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data


        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations


        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 40;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);

            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(testData);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testData.reset();
            trainData.reset();
        }

        log.info("----- Example Complete -----");
    }

    //下载数据集合
    private static void downloadUCIData() throws Exception {
        String tempDir = System.getProperty("java.io.tmpdir");
        String tmpDirectory = FilenameUtils.concat(tempDir, "brainhole/seqclassification/");
        baseDir = new File(tmpDirectory);
        baseTrainDir = new File(baseDir, "train");
        featuresDirTrain = new File(baseTrainDir, "features");
        labelsDirTrain = new File(baseTrainDir, "labels");
        baseTestDir = new File(baseDir, "test");
        featuresDirTest = new File(baseTestDir, "features");
        labelsDirTest = new File(baseTestDir, "labels");

        //初始化数据目录
        if (baseDir.exists()) {
            baseDir.deleteOnExit();    
        }
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        String file = "data/seqclassification/data.csv";
        FileReader fReader = new FileReader(file);
        CSVReader csvReader = new CSVReader(fReader); 
        List<String[]> rows = csvReader.readAll();
        
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String[] row : rows) {
            String[] features = new String[row.length-1];
            String[] labels = new String[1];
            System.arraycopy(row, 0, features, 0, features.length);
            System.arraycopy(row, features.length, labels, 0, labels.length);
            
            String transposed = String.join("\n", features);
            String labelposed = String.join("\n", labels);
            //标签: 每100行一个标签，标签从0开始依次加1
            contentAndLabels.add(new Pair<>(transposed, Integer.valueOf(labelposed)));
        }

        //随机打乱数据顺序
        Collections.shuffle(contentAndLabels, new Random(12345));
        
        //训练数据行数450行（总行数600行，拆分比例训练（75%），测试（25%））
        int nTrain = 450;   //75% train, 25% test
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels) {
            File outPathFeatures;
            File outPathLabels;
            if (trainCount < nTrain) {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            } else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }
            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }
    }
}
