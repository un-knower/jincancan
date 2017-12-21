package org.iplatform.microservices.brain.service.multiLayernetwork;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class NeuralNetworks {
    protected static Logger log = LoggerFactory.getLogger(NeuralNetworks.class);
    protected MultiLayerNetwork model;
    protected DataSet trainData;
    protected DataSet testData;
    protected Normalizer normalizer;

    public NeuralNetworks() {

    }

    public NeuralNetworks(MultiLayerNetwork model, Normalizer normalizer, IterationListener iterationListener,
            StatsListener statsListener) {
        this.model = model;
        this.model.setListeners(iterationListener);
        if (statsListener != null) {
            this.model.setListeners(statsListener);
        }
        this.normalizer = normalizer;
    }

    public File saveModel(String modelDirectory, String modelFile, boolean saveUpdater) throws IOException {
        File dirFile = new File(modelDirectory);
        dirFile.mkdirs();
        File locationToSave = new File(dirFile.getAbsolutePath()+File.separator+modelFile);
        ModelSerializer.writeModel(this.model, locationToSave, saveUpdater);
        log.info(String.format("模型存储 %s", locationToSave.getAbsolutePath()));
        return locationToSave;
    }
}
