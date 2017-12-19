package org.iplatform.microservices.brain.service.example.nlp;

import java.util.Collection;

import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyVectorsListener implements VectorsListener {
    private static Logger log = LoggerFactory.getLogger(MyVectorsListener.class);
    
    @Override
    public boolean validateEvent(ListenerEvent event, long argument) {
        return true;
    }

    @Override
    public void processEvent(ListenerEvent event, SequenceVectors sequenceVectors, long argument) {
//        log.info(String.format("score: %.8f", sequenceVectors.getSequencesScore()));
//        //Collection<VocabWord> vocabWords  = sequenceVectors.getVocab().vocabWords();
//        for(int i=0;i<sequenceVectors.getVocab().numWords();i++){
//            String word = sequenceVectors.getVocab().wordAtIndex(i);
//            int frequency = sequenceVectors.getVocab().wordFrequency(word);
//            log.info(String.format("[%s] 频度 %d", word,frequency));    
//        }        
    }
}
