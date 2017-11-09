package org.iplatform.microservices.brain.util;

import org.nd4j.linalg.api.ndarray.INDArray;

public class INDArrayUtil {
    
    public static int getMaxIndexFloatArrayFromSlice(INDArray rowSlice) {
        return maxIndex(getFloatArrayFromSlice(rowSlice));
    }

    /**
     * 找到数组中最大数字的位置，这个位置就是分类标示号
     *
     * @param vals
     * @return
     */
    private static int maxIndex(float[] vals) {
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++) {
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * 将INDArray对象转换为float数组
     * @param rowSlice
     * @return
     */
    private static float[] getFloatArrayFromSlice(INDArray rowSlice) {
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }
}
