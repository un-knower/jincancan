package org.iplatform.microservices.brain.service.example.animals;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 特征分类案例
 * 输入特征：年龄，食物，声音，重量
 * 输出特征：类别：猫/狗/人
 *
 * @author zhanglei
 */
public class AnimalsTest {
    private static Logger log = LoggerFactory.getLogger(AnimalsTest.class);
    private static Map<Integer, String> eats = readEnumCSV("data/animals/eats.csv");
    private static Map<Integer, String> sounds = readEnumCSV("data/animals/sounds.csv");
    private static Map<Integer, String> classifiers = readEnumCSV("data/animals/classifiers.csv");

    @Test
    public void test() {

        try {
            log.info("对数据拆分，生成训练数据和测试数据");
            int labelIndex = 4; //每行数据5个值，前面4个为输入特征，最后一个也就是下标为4的为答案             
            int numClasses = 3; //有3个分类，分别为0,1,2也就是上面的下标为4的取值范围
            int batchSizeTraining = 2000; //数据行数

            RecordReader recordReader = new CSVRecordReader(0, ",");
            recordReader.initialize(new FileSplit(new File("data/animals/animals_all.csv")));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSizeTraining, labelIndex,
                    numClasses);
            DataSet allData = iterator.next();
            allData.shuffle();
            //设置拆分比例为 训练数据（0.65）：测试数据（0.35）
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.45);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            //验证用数据
//            int batchSizePrediction = 3;
//            DataSet predictionData = readCSVDataset("data/animals/animals_prediction.csv", batchSizePrediction,
//                    labelIndex, numClasses);
            
            // 为每一条测试记录构造数据模型: 行号，map{年龄，食物，声音，重量}
            Map<Integer, Map<String, Object>> animals = makeAnimalsForTesting(testData);

            //对于不是0-1的要做标准化数据处理
            DataNormalization normalizer = new NormalizerStandardize();

            //获取 STDEV 基于样本估算标准偏差
            normalizer.fit(trainingData);

            //标准化训练数据，应该是要转为0-1之间的数据
            normalizer.transform(trainingData);
            //标准化测试数据，应该是要转为0-1之间的数据
            normalizer.transform(testData);
            //标准化校验数据，应该是要转为0-1之间的数据
            //normalizer.transform(predictionData);

            final int numInputs = 4;//输入特征数
            int outputNum = 3;//输出分类个数
            int iterations = 1000;//迭代次数
            long seed = 6;//随机因子
            int numEpochs = 10; //循环检查次数

            log.info("创建模型....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
                    .activation(Activation.TANH).weightInit(WeightInit.XAVIER).learningRate(0.1).regularization(true)
                    .l2(1e-4).list().layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                    .layer(2,
                            new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                    .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
                    .backprop(true).pretrain(false).build();

            //run the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            Evaluation eval = new Evaluation(3);
            INDArray output = model.output(testData.getFeatureMatrix());

            log.info("训练模型....");
            for (int i = 0; i < numEpochs; i++) {
                model.fit(trainingData);
                log.info("模型训练完成，使用测试数据进行模型评估");
                eval.eval(testData.getLabels(), output);
                log.info("-------------------------------------------------------");
                /*
                 * 精确率、召回率和F1值衡量的是模型的相关性。举例来说，“癌症不会复发”这样的预测结果（即假负例/假阴性）就有风险，
                 * 因为病人会不再寻求进一步治疗。所以，比较明智的做法是选择一种可以避免假负例的模型（即精确率、召回率和F1值较高），
                 * 尽管总体上的准确率可能会相对较低一些。
                 * */
                log.info("准确率:"+eval.accuracy());//模型准确识别出的MNIST图像数量占总数的百分比
                log.info("精确率:"+eval.precision());//真正例的数量除以真正例与假正例数之和
                log.info("召回率:"+eval.recall());//真正例的数量除以真正例与假负例数之和
                log.info("F1:"+eval.f1());//精确率和召回率的加权平均值
                log.info("-------------------------------------------------------");
            }

            log.info("根据测试数据预测分类，输出预测结果");
            setFittedClassifiers(output, animals);
            logAnimals(animals);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void logAnimals(Map<Integer, Map<String, Object>> animals) {
        for (Map<String, Object> a : animals.values())
            log.info(a.toString());//打印分类结果
    }

    public static void setFittedClassifiers(INDArray output, Map<Integer, Map<String, Object>> animals) {
        for (int i = 0; i < output.rows(); i++) {
            animals.get(i).put("classifier",
                    //classifiers是csv下标对照表，把分类id对应成名称,output.slice(i)=[0.00, 0.99, 0.01]，对应每种分类的得分，最高值即为分类
                    classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));
        }

    }

    /**
     * This method is to show how to convert the INDArray to a float array. This is to
     * provide some more examples on how to convert INDArray to types that are more java
     * centric.
     *
     * @param rowSlice
     * @return
     */
    public static float[] getFloatArrayFromSlice(INDArray rowSlice) {
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * find the maximum item index. This is used when the data is fitted and we
     * want to determine which class to assign the test row to
     *
     * @param vals
     * @return
     */
    public static int maxIndex(float[] vals) {
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++) {
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * take the dataset loaded for the matric and make the record model out of it so
     * we can correlate the fitted classifier to the record.
     *
     * @param testData
     * @return
     */
    public static Map<Integer, Map<String, Object>> makeAnimalsForTesting(DataSet testData) {
        Map<Integer, Map<String, Object>> animals = new HashMap<Integer, Map<String, Object>>();

        INDArray features = testData.getFeatureMatrix();
        for (int i = 0; i < features.rows(); i++) {
            INDArray slice = features.slice(i);
            Map<String, Object> animal = new HashMap();

            //set the attributes，对应输入的4个特征值
            animal.put("yearsLived", slice.getInt(0));
            //eats,sounds,从列表映射过来
            animal.put("eats", eats.get(slice.getInt(1)));
            animal.put("sounds", sounds.get(slice.getInt(2)));
            animal.put("weight", slice.getFloat(3));

            animals.put(i, animal);
        }
        return animals;

    }

    public static Map<Integer, String> readEnumCSV(String csvFileClasspath) {
        try {
            List<String> lines = IOUtils.readLines(new FileInputStream(csvFileClasspath));
            Map<Integer, String> enums = new HashMap<Integer, String>();
            for (String line : lines) {
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]), parts[1]);
            }
            return enums;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

    }

    /**
     * used for testing and training
     *
     * @param csvFileClasspath
     * @param batchSize
     * @param labelIndex
     * @param numClasses
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        return iterator.next();//一下子取出30个样本，也就是全部的数据了
    }

}
