/*
 * 前馈神经网络，卷积神经网络
 * 可用于分类预测
 * */
package org.iplatform.microservices.brain.service.multiLayernetwork;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.*;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.util.ModelSerializer;
import org.iplatform.microservices.brain.service.multiLayernetwork.listeners.LossScoreIterationListener;
import org.iplatform.microservices.brain.util.INDArrayUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author zhanglei
 * 
 */
public class FeedforwardNeuralNetworks extends NeuralNetworks {

    private static Logger log = LoggerFactory.getLogger(FeedforwardNeuralNetworks.class);

    private MultiLayerConfiguration multiLayerConfiguration;
    
    public FeedforwardNeuralNetworks(MultiLayerNetwork model, Normalizer normalizer, IterationListener iterationListener,
            StatsListener statsListener) {
        super(model,normalizer,iterationListener,statsListener);
    }
    
    /**
     * @param seed 随机因子
     * @param iterations 迭代次数
     * @param numInputs 数据列数
     * @param outputNum 输出分类数
     * @return
     */
    public FeedforwardNeuralNetworks(Normalizer normalizer, IterationListener iterationListener,
            StatsListener statsListener, long seed, int iterations, final int numInputs, int outputNum) {
        multiLayerConfiguration = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
                .activation(Activation.TANH).weightInit(WeightInit.XAVIER).learningRate(0.1).regularization(true)
                .l2(1e-4).list().layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(outputNum).build())
                .layer(1, new DenseLayer.Builder().nIn(outputNum).nOut(outputNum).build())
                .layer(2,
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX).nIn(outputNum).nOut(outputNum).build())
                .backprop(true).pretrain(false).build();
        this.model = new MultiLayerNetwork(multiLayerConfiguration);
        this.model.init();
        this.model.setListeners(iterationListener);
        if (statsListener != null) {
            this.model.setListeners(statsListener);
        }        
        this.normalizer = normalizer;
    }

    public boolean trainingCSV(File file, int skipNumLines, String delimiter, double percentTrain, int batchSize,
            int labelIndex, int numClasses, int numEpochs, double hopeScore,Boolean earlystop) throws IOException, InterruptedException {
        return this.trainingCSV(file, skipNumLines, delimiter, percentTrain, batchSize, labelIndex, numClasses,
                numEpochs, hopeScore, -1,earlystop);
    }

    public boolean trainingCSV(File file, int skipNumLines, String delimiter, double percentTrain, int batchSize,
                               int labelIndex, int numClasses, int numEpochs, double hopeScore, long timeoutMinute) throws IOException, InterruptedException {
        return this.trainingCSV(file, skipNumLines, delimiter, percentTrain, batchSize, labelIndex, numClasses, numEpochs, hopeScore, timeoutMinute,Boolean.FALSE);
    }

    public boolean trainingCSV(File file, int skipNumLines, String delimiter, double percentTrain, int batchSize,
            int labelIndex, int numClasses, int numEpochs, double hopeScore, long timeoutMinute, Boolean earlystop)
            throws IOException, InterruptedException {
        if (file.exists()) {
            RecordReader recordReader = new CSVRecordReader(skipNumLines, delimiter);
            recordReader.initialize(new FileSplit(file));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
            DataSet allData = iterator.next();
            allData.shuffle();
            SplitTestAndTrain splitTrain = allData.splitTestAndTrain(percentTrain);
            trainData = splitTrain.getTrain();
            testData = splitTrain.getTest();

            if(earlystop){
                return this.trainingEarlyStopping(trainData, testData, numClasses, numEpochs, hopeScore, timeoutMinute);
            }else{
                return this.training(trainData, testData, numClasses, numEpochs, hopeScore, timeoutMinute);
            }
        } else {
            log.error(String.format("文件 %s 不存在", file.getAbsolutePath()));
            return Boolean.FALSE;
        }
    }

    private boolean trainingEarlyStopping(DataSet trainingData, DataSet testData, int numClasses, int numEpochs, double hopeScore,
                             long timeoutMinute) {
        long s = System.currentTimeMillis();

        //获取 STDEV 基于样本估算标准偏差
        normalizer.fit(trainingData);
        //标准化训练数据，应该是要转为0-1之间的数据
        normalizer.transform(trainingData);
        //标准化测试数据，应该是要转为0-1之间的数据
        normalizer.transform(testData);

        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        File dirFile = new File(exampleDirectory);
        dirFile.mkdir();
        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .evaluateEveryNEpochs(1)
                .scoreCalculator(new DataSetLossCalculator(new TestDataSetIterator(testData), true)) //Calculate test set score
                .modelSaver(saver)
                .build();

        List<EpochTerminationCondition> epochTerminationConditions = new ArrayList();
        epochTerminationConditions.add(new MaxEpochsTerminationCondition(numEpochs));//循环次数终止设置
        epochTerminationConditions.add(new ScoreImprovementEpochTerminationCondition(100));//分值经过M个连续epoch没有改善时终止
        esConf.setEpochTerminationConditions(epochTerminationConditions);

        List<IterationTerminationCondition> iterationTerminationConditions = new ArrayList();
        if(timeoutMinute>-1) {
            //达到一定的时间上限时停止定型
            iterationTerminationConditions.add(new MaxTimeIterationTerminationCondition(timeoutMinute, TimeUnit.MINUTES));
        }
        if(hopeScore>-1){
            //分值超过一定数值便停止定型
            //iterationTerminationConditions.add(new MaxScoreIterationTerminationCondition(hopeScore));
        }
        esConf.setIterationTerminationConditions(iterationTerminationConditions);

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,multiLayerConfiguration,new TestDataSetIterator(trainingData));
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
        Collections.sort(list);
        this.model = result.getBestModel();
        long e = System.currentTimeMillis();

        log.info("+-----------------------------------------------------+");
        log.info("| Termination reason:" + result.getTerminationReason());//
        log.info("| Termination details:" + result.getTerminationDetails());//
        log.info("| Total epochs:" + result.getTotalEpochs());//
        log.info("| Best epoch number:" + result.getBestModelEpoch());//
        log.info("| Score at best epoch:" + result.getBestModelScore());//
        log.info("+-----------------------------------------------------+");
        log.info("| Epoch vs. Score");
        for( Integer i : list){
            log.info("| "+ i + "\t" + scoreVsEpoch.get(i));
        }
        log.info("+-----------------------------------------------------+");
        log.info("+ 耗时:" + (e - s) / 1000 + "秒");
        log.info("+-----------------------------------------------------+");
        return Boolean.TRUE;
    }
    private boolean training(DataSet trainingData, DataSet testData, int numClasses, int numEpochs, double hopeScore,
            long timeoutMinute) {
        //获取 STDEV 基于样本估算标准偏差
        normalizer.fit(trainingData);
        //标准化训练数据，应该是要转为0-1之间的数据
        normalizer.transform(trainingData);
        //标准化测试数据，应该是要转为0-1之间的数据
        normalizer.transform(testData);

        //超时时间
        long stopTime = System.currentTimeMillis() + (timeoutMinute * 60 * 1000);
        long numEpochsCounter = 0;
        log.info("开始训练模型");
        INDArray output;
        long s = System.currentTimeMillis();
        boolean succeed = Boolean.FALSE;
        Evaluation eval = new Evaluation(numClasses);
        while (System.currentTimeMillis() < stopTime || numEpochsCounter < numEpochs) {
            model.fit(trainingData);
            output = model.output(testData.getFeatureMatrix());
            //new Evaluation(numClasses);
            eval.eval(testData.getLabels(), output);
//            if (eval.f1() >= hopeScore) {
//                succeed = Boolean.TRUE;
//                break;
//            } else {
//                log.info("评价分:" + eval.f1() + ",期望分:" + hopeScore);
//            }
            log.info("epoch: "+numEpochsCounter+" \t评价分:" + eval.f1());
            numEpochsCounter++;
        }
        long e = System.currentTimeMillis();
        log.info("+-----------------------------------------------------+");
        if (succeed) {
            log.info("| 训练成功(期望评价分:" + hopeScore + ")");
        } else {
            log.info("| 训练失败(期望评价分:" + hopeScore + ")");
        }
        log.info("+-----------------------------------------------------+");
        /*
         * 精确率、召回率和F1值衡量的是模型的相关性。举例来说，“癌症不会复发”这样的预测结果（即假负例/假阴性）就有风险，
         * 因为病人会不再寻求进一步治疗。所以，比较明智的做法是选择一种可以避免假负例的模型（即精确率、召回率和F1值较高），
         * 尽管总体上的准确率可能会相对较低一些。
         *
         * 举例：
         * 真实数据：100人，20女生，80男生
         * 分类训练后得到结果：50女生（20真实女生，30男生当成了女生），男生50
         * 准确率(Accuracy)=(正确的女生数 + 正确的男生数) / 总人数 = (20 + 50) / 100 = 0.7
         * 精确率(Precision)=正确的女生数 / (正确的女生数 + 将男生当成了女生的数量) = 20 / (20 + 30) = 0.4
         * 召回率(Recall)=正确的女生数 / (正确的女生数 + 将女生当成了男生的数量) = 20 / (20 + 0)= 1
         * 评价分(F1)=(2 * 精准率 * 召回率)/(精准率 + 召回率) = (2 * 0.4 * 1) / (0.4 + 1) = 0.57
         *
         * 以上公式的F1值使用的精确值和召回率的调和均值，但是Deeplearning4j好像不是用的调和平均
         * */
        log.info("| 准确率(Accuracy):" + eval.accuracy());// 准确识别出来的数量/数据总数量
        log.info("| 精确率(Precision):" + eval.precision());// 准确正例的数量／(真正例数量+假正例数量)
        log.info("| 召回率(Recall):" + eval.recall());//真正例的数量/(真正例数量+假负例数量)
        log.info("| 评价分(F1):" + eval.f1());//精确率和召回率的加权平均值
        log.info("+-----------------------------------------------------+");
        log.info("+ 耗时:" + (e - s) / 1000 + "秒");
        log.info("+-----------------------------------------------------+");
        //return succeed;
        return Boolean.TRUE;
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
            String predictHeaderName = "";
            if (labelIndex > 0) {
                csvData.get(i).put(headerNames.get(labelIndex),
                        INDArrayUtil.getMaxIndexFloatArrayFromSlice(predictData.getLabels().slice(i)));
                csvData.get(i).put(headerNames.get(labelIndex) + "(预测值)",
                        INDArrayUtil.getMaxIndexFloatArrayFromSlice(predictionLabels.slice(i)));
            } else {
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
            int index = 0;
            for (String headerName : headerNames) {
                if (index < slice.length()) {
                    animal.put(headerName, slice.getInt(index++));
                } else {
                    animal.put(headerName, null);
                }
            }
            animals.put(i, animal);
        }
        return animals;
    }

    public static class FeedforwardNeuralNetworksBuilder {
        private long nestedSeed = 6;
        private int nestedIterations = 1000;
        private int nestedNumInputs;
        private int nestedOutputNum;
        private MultiLayerNetwork nestedModel;
        private IterationListener nestedIterationListener;
        private StatsListener nestedStatsListener;
        private Normalizer nestedNormalizer = new NormalizerStandardize();

        public FeedforwardNeuralNetworksBuilder(final int numInputs, final int outputNum) {
            this.nestedNumInputs = numInputs;
            this.nestedOutputNum = outputNum;
        }

        public FeedforwardNeuralNetworksBuilder seed(long seed) {
            this.nestedSeed = seed;
            return this;
        }

        public FeedforwardNeuralNetworksBuilder iterations(int iterations) {
            this.nestedIterations = iterations;
            return this;
        }

        public FeedforwardNeuralNetworksBuilder normalizer(Normalizer normalizer) {
            this.nestedNormalizer = normalizer;
            return this;
        }

        public FeedforwardNeuralNetworksBuilder loadModel(File modelFile) throws IOException {
            this.nestedModel = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            return this;
        }

        public FeedforwardNeuralNetworksBuilder iterationListener(IterationListener iterationListener) throws IOException {
            this.nestedIterationListener = iterationListener;
            return this;
        }

        public FeedforwardNeuralNetworksBuilder iterationListener(StatsListener statsListener) throws IOException {
            this.nestedStatsListener = statsListener;
            return this;
        }

        public FeedforwardNeuralNetworks build() {
            if (nestedIterationListener == null) {
                this.nestedIterationListener = new LossScoreIterationListener(100);
            }
            if (this.nestedModel != null) {
                return new FeedforwardNeuralNetworks(this.nestedModel, this.nestedNormalizer,
                        this.nestedIterationListener,this.nestedStatsListener);
            } else {
                return new FeedforwardNeuralNetworks(this.nestedNormalizer, this.nestedIterationListener,this.nestedStatsListener,
                        this.nestedSeed, this.nestedIterations, this.nestedNumInputs, this.nestedOutputNum);
            }
        }
    }

}
