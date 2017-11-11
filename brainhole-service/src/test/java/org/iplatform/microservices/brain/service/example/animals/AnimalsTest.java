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
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.iplatform.microservices.brain.service.multiLayernetwork.FeedforwardNeuralNetworks;
import org.junit.Test;
import org.junit.runner.RunWith;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * 特征分类案例
 * 输入特征：年龄，食物，声音，重量
 * 输出特征：类别：猫/狗/人
 *
 * @author zhanglei
 */
public class AnimalsTest {
    private static Logger log = LoggerFactory.getLogger(AnimalsTest.class);

    @Test
    public void test2() throws Exception, InterruptedException {
        int skipNumLines = 1;//cvs文件列头占用的行数
        String delimiter = ",";//cvs文件分隔符
        double percentTrain = 0.7;//文件拆分比例，训练集0.7,测试集0.3
        int batchSize = 148;//批次数，数量于cvs文件行数相等效果最好，但是比较满（结合分布式计算可以将一个大数据集分成多批由每个节点分别计算）
        int numInputs=4;//特征向量个数
        int labelIndex = 4;//结果列在cvs的位置
        int outputNum = 3;//结果分类数
        int numEpochs = 10;//循环次数，循环次数>cvs文件总行数/递归批次数
        double hopeScore = 0.97d;//期望评分数，当训练后的分数大于等于期望评分数后停止训练
        
        FeedforwardNeuralNetworks feedforwardNeuralNetworks = new FeedforwardNeuralNetworks.NetworkServiceBuilder(numInputs, outputNum).seed(6).iterations(1000).build();
        //训练模型
        boolean trainingSucceed = feedforwardNeuralNetworks.trainingCSV(new File("data/animals/animals_code.csv"), skipNumLines,
                delimiter, percentTrain, batchSize, labelIndex, outputNum, numEpochs, hopeScore);

        //保存模型
        if (trainingSucceed) {
            feedforwardNeuralNetworks.saveModel("model.brain", Boolean.TRUE);

            //使用模型进行分类预测
            int predictFileRowCount = 3;//预测数据3行

            //情况2，验证数据不包含实际值
            feedforwardNeuralNetworks.predict(new File("data/animals/animals_prediction.csv"), skipNumLines, delimiter,
                    predictFileRowCount);

            //情况1，验证数据包含实际值，可用于比较实际值和预测值
            feedforwardNeuralNetworks.predict(new File("data/animals/animals_prediction_compare.csv"), skipNumLines, delimiter,
                    predictFileRowCount, labelIndex, outputNum);
        } else {
            log.info("模型训练结束，未达到期望评价分:" + hopeScore);
        }
    }

}
