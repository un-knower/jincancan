/*
* 情感识别
* */
package org.iplatform.microservices.brain.service.example.nlp.sentiment;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerFactory.ChineseTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.iplatform.microservices.brain.service.example.nlp.MyCommonPreprocessor;
import org.iplatform.microservices.brain.service.example.nlp.MyVectorsListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Set;

public class SentimentRecognitionTest {

    private static Logger log = LoggerFactory.getLogger(SentimentRecognitionTest.class);

    @Test
    public void test() throws Exception {

        Word2Vec negWord2Vec = createWord2Vec("data/sentiment/neg.txt");
        Word2Vec posWord2Vec = createWord2Vec("data/sentiment/pos.txt");

        int batchSize = 64;     //Number of examples in each minibatch
        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this
        final int seed = 0;     //Seed for reproducibility
//        SentimentExampleIterator train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
//        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

//        log.info("词 \t 频度");
//        for (int i = 0; i < vec.getVocab().numWords(); i++) {
//            String word = vec.getVocab().wordAtIndex(i);
//            int frequency = vec.getVocab().wordFrequency(word);
//            log.info(String.format("%s \t %d", word, frequency));
//        }

        //词表的维度
        int VOCAB_SIZE = 256;

        //构建神经网络


        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(Updater.ADAM)  //To configure: .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(2e-2)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

    }

    private Word2Vec createWord2Vec(String sentimentFile) throws FileNotFoundException {
        String tempDir = System.getProperty("java.io.tmpdir");
        String modelDirectory = FilenameUtils.concat(tempDir, "brainhole/sentiment/");
        File dir = new File(modelDirectory);
        dir.mkdirs();
        String word2vcModel = modelDirectory + File.separator + "word2vc.txt";
        Word2Vec vec = null;
        if (new File(word2vcModel).exists()) {
            vec = WordVectorSerializer.readWord2VecModel(word2vcModel);
        } else {
            SentenceIterator lineiter = new BasicLineIterator(sentimentFile);
            TokenizerFactory t = new ChineseTokenizerFactory();
            t.setTokenPreProcessor(new MyCommonPreprocessor());
            vec = new Word2Vec.Builder().minWordFrequency(1) //词在文本中出现次数少于N次，则不予学习
                    .iterations(10) //网络在处理一批数据时允许更新系数的次数，迭代次数太少，网络可能来不及学习所有能学到的信息；迭代次数太多则会导致网络定型时间变长
                    .layerSize(500) //指定词向量中的特征数量
                    .seed(42) //随机因子
                    .windowSize(5)
                    .iterate(lineiter)
                    .tokenizerFactory(t).build();
            Set<VectorsListener<VocabWord>> sets = new HashSet();
            MyVectorsListener li = new MyVectorsListener();
            sets.add(li);
            vec.setEventListeners(sets);
            vec.fit();
            WordVectorSerializer.writeWord2VecModel(vec, word2vcModel);
        }
        return vec;
    }
}
