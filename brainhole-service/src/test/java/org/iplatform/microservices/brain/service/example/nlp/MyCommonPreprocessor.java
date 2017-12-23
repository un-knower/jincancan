package org.iplatform.microservices.brain.service.example.nlp;

import java.util.regex.Pattern;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

public class MyCommonPreprocessor implements TokenPreProcess {
    
    private static final Pattern punctPattern = Pattern.compile("[\\d\\.:,\"\'\\(\\)\\[\\]|/?!;，：“”。．！]+");
    
    @Override
    public String preProcess(String token) {
        if(token.length()>=2){
            return punctPattern.matcher(token).replaceAll("");
        }else{
            return "";
        }

    }
}
