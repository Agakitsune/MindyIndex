package org.example;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        MindyIndex index;

        File file = new File("model.zip");
        if (file.exists()) {
            System.out.println("Model already exists. Skipping training.");

            index = MindyIndex.loadIndex(file);
            index.loadStopWord("stopwords.txt")
                    .index("test.txt")
                    .index("test1.txt")
                    .index("test2.txt")
                    .fit();
        } else {
            System.out.println("Model does not exist. Training model.");

            index = new MindyIndex.Builder()
                    .loadStopWord("stopwords.txt")
                    .minWordFrequency(0)
                    .iterations(1_000)
                    .epochs(1)
                    .layerSize(1000)
                    .seed(10)
                    .windowSize(10)
                    .build();

            index.vectorize("rapture.txt")
                    .vectorize("fallout.txt")
                    .vectorize("helldiver.txt")
                    .vectorize("cyberpunk.txt")
                    .index("test.txt")
                    .index("test1.txt")
                    .index("test2.txt")
                    .fit();

            index.saveIndex("model.zip");
        }

        Scanner scan = new Scanner(System.in);
        while (true) {
            String input;
            try {
                System.out.print("Input: ");
                input = scan.nextLine();
            } catch (NoSuchElementException e) {
                System.out.println("exited");
                break;
            }
            Collection<String> lst = index.lookupNearest(10, input);
            Collection<String> indexs = index.lookupIndex(1, input);
            System.out.println("Closest Words to '" + input + "': " + lst);
            indexs.forEach(System.out::println);
        }
    }
}
