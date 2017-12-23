/*
* 情感识别
* */
package org.iplatform.microservices.brain.service.example.nlp.journeytothewest;

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
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerFactory.ChineseTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
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
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class JourneyToTheWestTest {

    private static Logger log = LoggerFactory.getLogger(JourneyToTheWestTest.class);

    @Test
    public void test() throws Exception {
        Word2Vec vec;
        String model = "data/books/红楼梦.model";
        if(new File(model).exists()){
            vec = WordVectorSerializer.readWord2VecModel(model);
        }else{
            vec = createWord2Vec("data/books/红楼梦.txt",5,10,500,40,5);
            WordVectorSerializer.writeWord2VecModel(vec, model);
        }


        vec.lookupTable().plotVocab(500,new File("plot.txt"));

//        log.info("===================================");
//        for(int i=0;i<vec.getVocab().numWords();i++){
//            String word = vec.getVocab().wordAtIndex(i);
//            int frequency = vec.getVocab().wordFrequency(word);
//            log.info(String.format("%s \t %d", word,frequency));
//        }

        String word = "贾母";
        Collection<String> nearestWords = vec.wordsNearest(word, 10);
        log.info("与["+word+"]相似性最高的10个词");
        log.info("===================================");
        for(String nearestWord : nearestWords){
            //输出余弦相似性
            double consine = vec.similarity(word,nearestWord);
            log.info(String.format("%s \t %f",nearestWord,consine));
        }

    }

    private Word2Vec createWord2Vec(String sentimentFile,int minWordFrequency,int iterations, int layerSize,int seed,int windowSize) throws FileNotFoundException {
        long s = System.currentTimeMillis();
        SentenceIterator lineiter = new BasicLineIterator(sentimentFile);
        TokenizerFactory t = new ChineseTokenizerFactory();
        t.setTokenPreProcessor(new MyCommonPreprocessor());
        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(minWordFrequency) //词在文本中出现次数少于N次，则不予学习
                .iterations(iterations) //网络在处理一批数据时允许更新系数的次数，迭代次数太少，网络可能来不及学习所有能学到的信息；迭代次数太多则会导致网络定型时间变长
                .layerSize(layerSize) //指定词向量中的特征数量
                .seed(seed) //随机因子
                .windowSize(windowSize)
                .iterate(lineiter)
                .tokenizerFactory(t).build();
        Set<VectorsListener<VocabWord>> sets = new HashSet();
        MyVectorsListener li = new MyVectorsListener();
        sets.add(li);
        vec.setEventListeners(sets);
        vec.fit();
        long e = System.currentTimeMillis();
        log.info(String.format("Word2Vec 训练耗时 %s 秒",(e-s)/1000));
        return vec;
    }
}
