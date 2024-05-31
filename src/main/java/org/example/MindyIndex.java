package org.example;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.AggregatingSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

public class MindyIndex {
    private @NotNull Word2Vec model;
    private SentenceIterator iterator = null;
    private final List<String> stopWords = new ArrayList<>();
    private int iterations = 0;
    private int epochs = 0;
    private long seed = 0;
    private final List<String> unindexed = new ArrayList<>();
    private final Map<INDArray, String> indexs = new HashMap<>();

    private MindyIndex(@NotNull Word2Vec model) {
        this.model = model;
    }

    public MindyIndex vectorize(@NotNull String filePath) throws FileNotFoundException {
        return vectorize(new FileInputStream(filePath));
    }

    public MindyIndex vectorize(@NotNull File file) throws FileNotFoundException {
        return vectorize(new FileInputStream(file));
    }

    public MindyIndex vectorize(@NotNull InputStream stream) {
        if (this.iterator == null) {
            this.iterator = new BasicLineIterator(stream);
        } else {
            this.iterator = new AggregatingSentenceIterator.Builder()
                    .addSentenceIterator(this.iterator)
                    .addSentenceIterator(new BasicLineIterator(stream))
                    .build();
        }

        // this.model.getStopWords();
        this.model = new Word2Vec.Builder()
                .vocabCache(this.model.getVocab())
                .lookupTable(this.model.getLookupTable())
                .modelUtils(this.model.getModelUtils())
                .stopWords(this.stopWords)
                .minWordFrequency(model.getMinWordFrequency())
                .iterations(this.iterations)
                .epochs(this.epochs)
                .layerSize(model.getLayerSize())
                .seed(this.seed)
                .windowSize(model.getWindow())
                .iterate(this.iterator)
                .tokenizerFactory(this.model.getTokenizerFactory())
                .build();

        return this;
    }

    public MindyIndex vectorizePaths(@NotNull Collection<String> filePaths) {
        return vectorizeStreams(filePaths.stream().map(path -> {
            try {
                return new FileInputStream(path);
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }).collect(Collectors.toList()));
    }

    public MindyIndex vectorizeFiles(@NotNull Collection<File> files) {
        return vectorizeStreams(files.stream().map(file -> {
            try {
                return new FileInputStream(file);
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }).collect(Collectors.toList()));
    }

    public MindyIndex vectorizeStreams(@NotNull Collection<InputStream> streams) {
        for (InputStream stream : streams)
            vectorize(stream);
        return this;
    }

    public MindyIndex index(@NotNull String filePath) throws IOException {
        return index(new FileInputStream(filePath));
    }

    public MindyIndex index(@NotNull File file) throws IOException {
        return index(new FileInputStream(file));
    }

    public MindyIndex index(@NotNull InputStream stream) throws IOException {
        final int available = stream.available();
        if (available == 0) {
            return this;
        }
        byte[] bytes = new byte[available];
        final int red = stream.read(bytes);
        if (available != red) {
            return this;
        }
        String text = new String(bytes, StandardCharsets.UTF_8);
        unindexed.add(text);
        return vectorize(stream);
    }

    public MindyIndex indexData(@NotNull String data) {
        unindexed.add(data);
        return vectorize(new ByteArrayInputStream(data.getBytes(StandardCharsets.UTF_8)));
    }

    public MindyIndex fit() {
        this.model.fit();

        for (String str : unindexed) {
            indexs.put(vectorizeInput(str), str);
        }
        unindexed.clear();

        return this;
    }

    public MindyIndex loadStopWord(@NotNull String path) throws FileNotFoundException {
        return loadStopWord(new FileInputStream(path));
    }
    public MindyIndex loadStopWord(@NotNull File file) throws FileNotFoundException {
        return loadStopWord(new FileInputStream(file));
    }
    public MindyIndex loadStopWord(@NotNull InputStream stream) {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));

        reader.lines().forEach(
                str -> stopWords.add(str.trim())
        );

        return this;
    }

    public MindyIndex addStopWords(@NotNull Collection<String> words) {
        stopWords.addAll(words);
        return this;
    }

    public MindyIndex setStopWords(@NotNull Collection<String> words) {
        stopWords.clear();
        return addStopWords(words);
    }

    private INDArray vectorizeInput(@NotNull String input) {
        TokenizerFactory t = model.getTokenizerFactory();

        List<String> tokens = t.create(input).getTokens();

        INDArray vector = Nd4j.create(1, model.getLayerSize());

        for (String token : tokens) {
            INDArray wordVector = model.getWordVectorMatrix(token);
            if (wordVector != null) {
                vector.addi(wordVector);
            }
        }

        return vector;
    }

    public Collection<String> lookupNearest(int n, @NotNull String input) {
        return model.wordsNearest(vectorizeInput(input), n);
    }

    public Collection<String> lookupIndex(int n, @NotNull String input) {
        INDArray vector = vectorizeInput(input);
        List<String> lst = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            lst.add("");
        }
        double[] sims = new double[n];

        indexs.forEach((indArray, s) -> {
            double sim = Transforms.cosineSim(vector, indArray);
            for (int i = 0; i < n; i++) {
                if (sim > sims[i]) {
                    for (int j = n - 1; j > i; j--) {
                        sims[j] = sims[j - 1];
                        lst.set(j, lst.get(j - 1));
                    }
                    sims[i] = sim;
                    lst.set(i, s);
                    break;
                }
            }
        });

        return lst;
    }

    public static MindyIndex loadIndex(@NotNull String path) {
        return new MindyIndex(WordVectorSerializer.readWord2VecModel(path));
    }

    public static MindyIndex loadIndex(@NotNull File file) {
        return new MindyIndex(WordVectorSerializer.readWord2VecModel(file));
    }

    public void saveIndex(@NotNull String path) throws IOException {
        saveIndex(new FileOutputStream(path));
    }
    public void saveIndex(@NotNull File file) throws IOException {
        saveIndex(new FileOutputStream(file));
    }
    public void saveIndex(@NotNull OutputStream stream) throws IOException {
        WordVectorSerializer.writeWord2VecModel(model, stream);
    }

    public static class Builder {
        private int minWordFrequency = 0;
        private int iterations = 0;
        private int epochs = 0;
        private int layerSize = 0;
        private long seed = 0;
        private int windowSize = 0;
        private final Collection<InputStream> data = new ArrayList<>();
        private final List<String> stopWords = new ArrayList<>();
        private TokenizerFactory factory = new DefaultTokenizerFactory();

        public Builder() {
            factory.setTokenPreProcessor(new MindyPreprocessor());
        }

        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder windowSize(int size) {
            this.windowSize = size;
            return this;
        }

        public Builder tokenizer(@NotNull TokenizerFactory factory) {
            this.factory = factory;
            return this;
        }

        public Builder loadStopWord(@NotNull String path) throws FileNotFoundException {
            return loadStopWord(new FileInputStream(path));
        }
        public Builder loadStopWord(@NotNull File file) throws FileNotFoundException {
            return loadStopWord(new FileInputStream(file));
        }

        public Builder loadStopWord(@NotNull InputStream stream) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));

            reader.lines().forEach(
                    str -> stopWords.add(str.trim())
            );

            return this;
        }

        public Builder vectorize(@NotNull String filePath) throws FileNotFoundException {
            return vectorize(new FileInputStream(filePath));
        }

        public Builder vectorize(@NotNull File file) throws FileNotFoundException {
            return vectorize(new FileInputStream(file));
        }

        public Builder vectorize(@NotNull InputStream stream) {
            this.data.add(stream);
            return this;
        }

        public MindyIndex build() {
            Word2Vec.Builder builder = new Word2Vec.Builder()
                    .stopWords(stopWords)
                    .minWordFrequency(this.minWordFrequency)
                    .iterations(this.iterations)
                    .epochs(this.epochs)
                    .layerSize(this.layerSize)
                    .seed(this.seed)
                    .windowSize(this.windowSize)
                    .tokenizerFactory(this.factory);

            Collection<SentenceIterator> iterators = new ArrayList<>(data.size());
            for (InputStream stream : data) {
                iterators.add(new BasicLineIterator(stream));
            }

            Word2Vec model = null;

            if (!iterators.isEmpty()) {
                SentenceIterator sentenceIterator = new AggregatingSentenceIterator.Builder()
                        .addSentenceIterators(iterators)
                        .build();

                builder.iterate(sentenceIterator);

                model = builder.build();
                model.fit();
            } else {
                model = builder.build();
            }

            MindyIndex index = new MindyIndex(model);

            index.setStopWords(stopWords);
            index.seed = this.seed;
            index.iterations = this.iterations;
            index.epochs = this.epochs;

            return index;
        }
    }

}
