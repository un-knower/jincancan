package org.iplatform.microservices.brain.service.service;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FilenameUtils;
import org.iplatform.microservices.brain.service.multiLayernetwork.FeedforwardNeuralNetworks;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author zhanglei
 *
 */
@Configuration
@Service
@RestController
@RequestMapping("/api/v1")
public class APIService {
    private static final Logger logger = LoggerFactory.getLogger(APIService.class);


    @Autowired
    NeuralNetworksMonitorService neuralNetworksMonitorListener;

    @RequestMapping(value = "/hi", method = RequestMethod.GET)
    @ResponseBody
    public String hello() {
        return "Hi I'm Brain Hole";
    }

    @RequestMapping(value = "/animals", method = RequestMethod.GET)
    @ResponseBody
    public String animalsTest() throws IOException, InterruptedException {
        int skipNumLines = 1;//cvs文件列头占用的行数
        String delimiter = ",";//cvs文件分隔符
        double percentTrain = 0.7;//文件拆分比例，训练集0.7,测试集0.3
        int batchSize = 148;//批次数，数量于cvs文件行数相等效果最好，但是比较满（结合分布式计算可以将一个大数据集分成多批由每个节点分别计算）
        int numInputs = 4;//特征向量个数
        int labelIndex = 4;//结果列在cvs的位置
        int outputNum = 3;//结果分类数
        int numEpochs = 10;//循环次数，循环次数>cvs文件总行数/递归批次数
        double hopeScore = 0.97d;//期望评分数，当训练后的分数大于等于期望评分数后停止训练

        FeedforwardNeuralNetworks feedforwardNeuralNetworks = new FeedforwardNeuralNetworks.FeedforwardNeuralNetworksBuilder(
                numInputs, outputNum).seed(6).iterations(1000).iterationListener(neuralNetworksMonitorListener.getStatsListener()).build();
        //训练模型
        boolean trainingSucceed = feedforwardNeuralNetworks.trainingCSV(new File("data/fnn/animals_code.csv"),
                skipNumLines, delimiter, percentTrain, batchSize, labelIndex, outputNum, numEpochs, hopeScore,false);

        //保存模型
        if (trainingSucceed) {
            String tempDir = System.getProperty("java.io.tmpdir");
            String modelDirectory = FilenameUtils.concat(tempDir, "brainhole/fnn/");
            feedforwardNeuralNetworks.saveModel(modelDirectory,"model.brain", Boolean.TRUE);

            //使用模型进行分类预测
            int predictFileRowCount = 3;//预测数据3行

            //情况2，验证数据不包含实际值
            feedforwardNeuralNetworks.predict(new File("data/fnn/animals_prediction.csv"), skipNumLines, delimiter,
                    predictFileRowCount);

            //情况1，验证数据包含实际值，可用于比较实际值和预测值
            feedforwardNeuralNetworks.predict(new File("data/fnn/animals_prediction_compare.csv"), skipNumLines,
                    delimiter, predictFileRowCount, labelIndex, outputNum);
        } else {
            logger.info("模型训练结束，未达到期望评价分:" + hopeScore);
        }
        return "OK";
    }
}
