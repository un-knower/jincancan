package org.iplatform.microservices.brain.service.multiLayernetwork;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.iplatform.microservices.brain.service.multiLayernetwork.listeners.ScoreIterationListener;
import org.iplatform.microservices.brain.util.INDArrayUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * @author zhanglei
 *
 */
@Service
public class NetworkService {

    private static Logger log = LoggerFactory.getLogger(NetworkService.class);
    private MultiLayerNetwork model;
    private DataSet trainData;
    private DataSet testData;
    private DataSet predictionData;
    private Normalizer normalizer;

    public NetworkService(MultiLayerNetwork model, Normalizer normalizer, IterationListener iterationListener) {
        this.model = model;
        this.model.setListeners(iterationListener);
        this.normalizer = normalizer;
    }

    /**
     * @param seed 随机因子
     * @param iterations 迭代次数
     * @param numInputs 数据列数
     * @param outputNum 输出分类数
     * @return
     */
    public NetworkService(Normalizer normalizer, IterationListener iterationListener, long seed, int iterations,
            final int numInputs, int outputNum) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
                .activation(Activation.TANH).weightInit(WeightInit.XAVIER).learningRate(0.1).regularization(true)
                .l2(1e-4).list().layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(outputNum).build())
                .layer(1, new DenseLayer.Builder().nIn(outputNum).nOut(outputNum).build())
                .layer(2,
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX).nIn(outputNum).nOut(outputNum).build())
                .backprop(true).pretrain(false).build();
        this.model = new MultiLayerNetwork(conf);
        this.model.init();
        this.model.setListeners(iterationListener);
        this.normalizer = normalizer;
    }
    
    public boolean trainingCSV(File file, int skipNumLines, String delimiter, double percentTrain, int batchSize,
            int labelIndex, int numClasses, int numEpochs, double hopeScore) throws IOException, InterruptedException {
        return this.trainingCSV(file, skipNumLines, delimiter, percentTrain, batchSize, labelIndex, numClasses, numEpochs, hopeScore, -1);
    }

    public boolean trainingCSV(File file, int skipNumLines, String delimiter, double percentTrain, int batchSize,
            int labelIndex, int numClasses, int numEpochs, double hopeScore, long timeoutMinute) throws IOException, InterruptedException {
        if (file.exists()) {
            RecordReader recordReader = new CSVRecordReader(skipNumLines, delimiter);
            recordReader.initialize(new FileSplit(file));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
            DataSet allData = iterator.next();
            allData.shuffle();
            SplitTestAndTrain splitTrain = allData.splitTestAndTrain(percentTrain);
            trainData = splitTrain.getTrain();
            testData = splitTrain.getTest();
            SplitTestAndTrain splitTest = testData.splitTestAndTrain(0.5);
            testData = splitTest.getTrain();
            predictionData = splitTest.getTest();
            this.training(trainData, testData, predictionData, numClasses, numEpochs, hopeScore, timeoutMinute);
            return Boolean.TRUE;
        } else {
            log.error(String.format("文件 %s 不存在", file.getAbsolutePath()));
            return Boolean.FALSE;
        }
    }

    private boolean training(DataSet trainingData, DataSet testData, DataSet predictionData, int numClasses, int numEpochs,
            double hopeScore, long timeoutMinute) {
        //获取 STDEV 基于样本估算标准偏差
        normalizer.fit(trainingData);
        //标准化训练数据，应该是要转为0-1之间的数据
        normalizer.transform(trainingData);
        //标准化测试数据，应该是要转为0-1之间的数据
        normalizer.transform(testData);
        //标准化校验数据，应该是要转为0-1之间的数据
        normalizer.transform(predictionData);

        //超时时间
        long stopTime = System.currentTimeMillis()+(timeoutMinute*60*1000);
        long numEpochsCounter = 0;
        log.info("开始训练模型");
        INDArray output = null;
        long s = System.currentTimeMillis();
        while(System.currentTimeMillis()<stopTime || numEpochsCounter<numEpochs){
        //for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
            Evaluation eval = new Evaluation(numClasses);
            output = model.output(predictionData.getFeatureMatrix());
            eval.eval(predictionData.getLabels(), output);
            if (eval.f1() >= hopeScore) {
                long e = System.currentTimeMillis();
                log.info("+-----------------------------------------------------+");
                log.info("|               训练完成(期望评价分:" + hopeScore + ")");
                log.info("+-----------------------------------------------------+");
                /*
                 * 精确率、召回率和F1值衡量的是模型的相关性。举例来说，“癌症不会复发”这样的预测结果（即假负例/假阴性）就有风险，
                 * 因为病人会不再寻求进一步治疗。所以，比较明智的做法是选择一种可以避免假负例的模型（即精确率、召回率和F1值较高），
                 * 尽管总体上的准确率可能会相对较低一些。
                 * */
                log.info("| 准确率:" + eval.accuracy());//模型准确识别出的MNIST图像数量占总数的百分比
                log.info("| 精确率:" + eval.precision());//真正例的数量除以真正例与假正例数之和
                log.info("| 召回率:" + eval.recall());//真正例的数量除以真正例与假负例数之和
                log.info("| 评价分:" + eval.f1());//精确率和召回率的加权平均值
                log.info("+-----------------------------------------------------+");
                log.info("+ 耗时:"+(e-s)/1000+"秒");
                log.info("+-----------------------------------------------------+");
                return Boolean.TRUE;
            } else {
                log.info("评价分:" + eval.f1() + ",期望分:" + hopeScore);
            }
            numEpochsCounter++;
        }
        return Boolean.FALSE;
    }

    public void predict(File file, int skipNumLines, String delimiter, int batchSize)
            throws IOException, InterruptedException {
        this.predict(file, skipNumLines, delimiter, batchSize, -1, 0);
    }

    public void predict(File file, int skipNumLines, String delimiter, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException {
        List<String> headerNames = new ArrayList();
        if (skipNumLines > 0) {
            //读取列头
            RecordReader headReader = new CSVRecordReader(skipNumLines - 1, delimiter);
            try {
                headReader.initialize(new FileSplit(file));
                List<Writable> heads = headReader.next();
                for (Writable head : heads) {
                    headerNames.add(head.toString());
                }
            } finally {
                headReader.close();
            }
        }

        RecordReader recordReader = new CSVRecordReader(skipNumLines, delimiter);
        recordReader.initialize(new FileSplit(file));
        DataSetIterator iterator;
        if (labelIndex > 0) {
            iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        } else {
            iterator = new RecordReaderDataSetIterator(recordReader, batchSize);
        }

        DataSet predictData = iterator.next();

        Map<Integer, Map<String, Object>> csvData = makeCsvData(headerNames, predictData);

        normalizer.transform(predictData);
        INDArray predictionLabels = this.model.output(predictData.getFeatureMatrix());
        for (int i = 0; i < predictionLabels.rows(); i++) {
            String predictHeaderName="";
            if (labelIndex > 0) {
                csvData.get(i).put(headerNames.get(labelIndex),
                        INDArrayUtil.getMaxIndexFloatArrayFromSlice(predictData.getLabels().slice(i)));
                csvData.get(i).put(headerNames.get(labelIndex)+"(预测值)", INDArrayUtil.getMaxIndexFloatArrayFromSlice(predictionLabels.slice(i)));                
            }else{
                csvData.get(i).put("预测值", INDArrayUtil.getMaxIndexFloatArrayFromSlice(predictionLabels.slice(i)));
            }
            
            log.info(csvData.get(i).toString());
        }
    }

    private Map<Integer, Map<String, Object>> makeCsvData(List<String> headerNames, DataSet predictionData) {
        Map<Integer, Map<String, Object>> animals = new LinkedHashMap<Integer, Map<String, Object>>();

        INDArray features = predictionData.getFeatureMatrix();
        for (int i = 0; i < features.rows(); i++) {
            INDArray slice = features.slice(i);
            Map<String, Object> animal = new LinkedHashMap();
            int index=0;
            for(String headerName : headerNames) {    
                if (index < slice.length()) {
                    animal.put(headerName, slice.getInt(index++));
                }else{
                    animal.put(headerName, null); 
                }
            }
            animals.put(i, animal);
        }
        return animals;
    }

    public File saveModel(String modelFile, boolean saveUpdater) throws IOException {
        File locationToSave = new File(modelFile);
        ModelSerializer.writeModel(this.model, locationToSave, saveUpdater);
        log.info(String.format("模型存储 %s", locationToSave.getAbsolutePath()));
        return locationToSave;
    }

    public static class NetworkServiceBuilder {
        private long nestedSeed = 6;
        private int nestedIterations = 1000;
        private int nestedNumInputs;
        private int nestedOutputNum;
        private MultiLayerNetwork nestedModel;
        private IterationListener nestedIterationListener;
        private Normalizer nestedNormalizer = new NormalizerStandardize();

        public NetworkServiceBuilder(final int numInputs, final int outputNum) {
            this.nestedNumInputs = numInputs;
            this.nestedOutputNum = outputNum;
        }

        public NetworkServiceBuilder seed(long seed) {
            this.nestedSeed = seed;
            return this;
        }

        public NetworkServiceBuilder iterations(int iterations) {
            this.nestedIterations = iterations;
            return this;
        }

        public NetworkServiceBuilder normalizer(Normalizer normalizer) {
            this.nestedNormalizer = normalizer;
            return this;
        }

        public NetworkServiceBuilder loadModel(File modelFile) throws IOException {
            this.nestedModel = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            return this;
        }

        public NetworkServiceBuilder iterationListener(IterationListener iterationListener) throws IOException {
            this.nestedIterationListener = iterationListener;
            return this;
        }

        public NetworkService build() {
            if (nestedIterationListener == null) {
                this.nestedIterationListener = new ScoreIterationListener(100);
            }
            if (this.nestedModel != null) {
                return new NetworkService(this.nestedModel, this.nestedNormalizer, this.nestedIterationListener);
            } else {
                return new NetworkService(this.nestedNormalizer, this.nestedIterationListener, this.nestedSeed,
                        this.nestedIterations, this.nestedNumInputs, this.nestedOutputNum);
            }
        }
    }

}
