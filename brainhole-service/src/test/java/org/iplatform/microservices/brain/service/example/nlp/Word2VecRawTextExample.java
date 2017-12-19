package org.iplatform.microservices.brain.service.example.nlp;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerFactory.ChineseTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Word2vec是一个用于处理文本的双层神经网络。它的输入是文本语料，输出则是一组向量
 * 因为这些数据都是与词语相似的离散状态，而我们的目的只是求取这些状态之间的转移概率，即它们共同出现的可能性
 * 本教程就将介绍怎样为任何一组离散且同时出现的状态创建神经向量
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Word2VecRawTextExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

    public static void main(String[] args) throws Exception {

        //String filePath = "data/raw_sentences.txt";
        String filePath="data/nlp/红楼梦.txt";

        log.info("Load & Vectorize Sentences....");
        //数据迭代器，每次返回一行字符串   
        SentenceIterator iter = new BasicLineIterator(filePath);
        
        //定义一个分词器，在每行中按空格分词
        //TokenizerFactory t = new DefaultTokenizerFactory();
        TokenizerFactory t = new ChineseTokenizerFactory();

        //CommonPreprocessor默认词处理类，会替换词中的[\d\.:,"'\(\)\[\]|/?!;]+为空，同时将词转换为小写
        t.setTokenPreProcessor(new MyCommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5) //词在文本中出现次数少于5次，则不予学习
                .iterations(1) //网络在处理一批数据时允许更新系数的次数
                .layerSize(100) //指定词向量中的特征数量与特征空间的维度数量相
                .seed(42) //随机因子
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t).build();
        
        Set<VectorsListener<VocabWord>> sets = new HashSet();
        MyVectorsListener li = new MyVectorsListener();
        sets.add(li);
        vec.setEventListeners(sets);
        
        log.info("训练模型....");
        vec.fit();
        
        for(int i=0;i<vec.getVocab().numWords();i++){
            String word = vec.getVocab().wordAtIndex(i);
            int frequency = vec.getVocab().wordFrequency(word);
            //log.info(String.format("[%s] 频度 %d", word,frequency));
            System.out.println(String.format("%s,%d", word,frequency));
        } 
        

        String nearWord = "薛宝钗";
        Collection<String> lst = vec.wordsNearest(nearWord, 10);
        System.out.println("输出10个与"+nearWord+"相近的词: " + lst);
    }
}