package org.example;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

import java.util.regex.Pattern;

public class MindyPreprocessor implements TokenPreProcess {

    private static final Pattern punctPattern = Pattern.compile("[\\.:,\"'\\(\\)\\[\\]|\\/?!;”“]+");

    @Override
    public String preProcess(String token) {
        return punctPattern.matcher(token).replaceAll("");
    }
}
