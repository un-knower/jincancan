package org.iplatform.microservices.brain.service.multiLayernetwork.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Score iteration listener
 *
 * @author zhanglei
 */
public class ScoreIterationListener implements IterationListener {
    private int printIterations = 10;
    private static final Logger log = LoggerFactory.getLogger(ScoreIterationListener.class);
    private boolean invoked = false;
    private long iterCount = 0;

    /**
     * @param printIterations    frequency with which to print scores (i.e., every printIterations parameter updates)
     */
    public ScoreIterationListener(int printIterations) {
        this.printIterations = printIterations;
    }

    /** Default constructor printing every 10 iterations */
    public ScoreIterationListener() {
    }

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        this.invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if (printIterations <= 0)
            printIterations = 1;
        if (iterCount % printIterations == 0) {
            invoke();
            double result = model.score();
            log.info(String.format("第 %d 次迭代的分数:%f", iterCount,result));
        }
        iterCount++;
    }
}