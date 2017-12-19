package org.iplatform.microservices.brain.service.multiLayernetwork;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/*
 * 递归神经网络（RNN）
 * 可用于时间序列、笔迹识别、语音识别、日志分析、欺诈检测、网络安全
 * */
public class RecurrentNeuralNetworks extends NeuralNetworks {
    /**
     * @param seed 随机因子
     * @param iterations 迭代次数
     * @param learningRate 误差梯度
     * @param numInputs 数据列数
     * @param outputNum 输出分类数
     * @return
     */
    public RecurrentNeuralNetworks(Normalizer normalizer, IterationListener iterationListener,
            StatsListener statsListener, long seed, int iterations, float learningRate, final int numInputs, int outputNum) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(learningRate)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation("tanh").nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax").nIn(10).nOut(outputNum).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        this.model.setListeners(iterationListener);
        if (statsListener != null) {
            this.model.setListeners(statsListener);
        }
        this.normalizer = normalizer;
    }
}
