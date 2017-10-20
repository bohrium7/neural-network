package neuralNetwork;

import cern.colt.matrix.DoubleMatrix2D;

/**
 * @author Jason Chan
 *
 * @created Apr 3, 2017 2:58:59 PM
 *
 */
public class Image {
    public DoubleMatrix2D image,
                          label;
    
    public Image(DoubleMatrix2D img, DoubleMatrix2D lb) {
        image = img;
        label = lb;
    }
}

