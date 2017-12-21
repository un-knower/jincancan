package org.iplatform.microservices.brain.service.example.fnn;

import java.io.File;

import org.iplatform.microservices.brain.service.multiLayernetwork.FeedforwardNeuralNetworks;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author zhanglei
 * 
 * 训练效果目前不理想,以下是我训练出来的最好分数（训练集运行50次）
 * 准确率:0.57675
 * 精确率:0.3863201266526417
 * 召回率:0.34286541620867755
 * 评价分:0.36329796253170715
 * 
 * 可能原因：
 * 1.样本不平衡，每一类对应的样本的个数不同（可考虑采用过采样方式，复制个数少的数据使之达到每类平衡，这只是建议）
 * 2.分类样本缺失，quality没有出现0，1，2，9，10
 */
public class WineQualityTest {
    private static Logger log = LoggerFactory.getLogger(WineQualityTest.class);

    @Test
    public void testWineQuality() throws Exception, InterruptedException {
        int skipNumLines = 1;//cvs文件列头占用的行数
        String delimiter = ";";//cvs文件分隔符
        double percentTrain = 0.65;//文件拆分比例，训练集0.65,测试集0.35
        int batchSize = 1600;//每批次训练个数        
        int numEpochs = 50;//运行训练集的次数
        int labelIndex = 11;//结果列在cvs的位置
        int numInputs=11;//特征向量个数
        int outputNum = 10;//结果分类数        
        double hopeScore = 0.97d;//期望评分数，当训练后的分数大于等于期望评分数后停止训练
        
        //训练模型
        FeedforwardNeuralNetworks feedforwardNeuralNetworks = new FeedforwardNeuralNetworks.FeedforwardNeuralNetworksBuilder(numInputs, outputNum).seed(6).iterations(100).build();
        boolean trainingSucceed = feedforwardNeuralNetworks.trainingCSV(new File("data/winequality/winequality-red.csv"), skipNumLines,
                delimiter, percentTrain, batchSize, labelIndex, outputNum, numEpochs, hopeScore,false);

        //保存模型
        if (trainingSucceed) {
            log.info("模型训练成功");
        } else {
            log.info("模型训练失败");
        }
    }

    @Test
    public void testWineQualityEarlyStop() throws Exception, InterruptedException {
        int skipNumLines = 1;//cvs文件列头占用的行数
        String delimiter = ";";//cvs文件分隔符
        double percentTrain = 0.65;//文件拆分比例，训练集0.65,测试集0.35
        int batchSize = 1600;//每批次训练个数
        int numEpochs = 50;//运行训练集的次数
        int labelIndex = 11;//结果列在cvs的位置
        int numInputs=11;//特征向量个数
        int outputNum = 10;//结果分类数
        double hopeScore = 0.97d;//期望评分数，当训练后的分数大于等于期望评分数后停止训练

        //训练模型
        FeedforwardNeuralNetworks feedforwardNeuralNetworks = new FeedforwardNeuralNetworks.FeedforwardNeuralNetworksBuilder(numInputs, outputNum).seed(6).iterations(100).build();
        boolean trainingSucceed = feedforwardNeuralNetworks.trainingCSV(new File("data/winequality/winequality-red.csv"), skipNumLines,
                delimiter, percentTrain, batchSize, labelIndex, outputNum, numEpochs, hopeScore,true);

        //保存模型
        if (trainingSucceed) {
            log.info("模型训练成功");
        } else {
            log.info("模型训练失败");
        }
    }
}
