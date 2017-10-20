/**
 * 
 */
package neuralNetwork;

import java.util.List;

import cern.colt.matrix.DoubleMatrix2D;

/**
 * @author Jason Chan
 *
 * @created Apr 3, 2017 5:04:59 PM
 *
 */
public class MatrixPair {
	List<DoubleMatrix2D> 	nabla_b,
							nabla_w;

	public MatrixPair(List<DoubleMatrix2D> nb, List<DoubleMatrix2D> nw) {
		nabla_b = nb;
		nabla_w = nw;
	}
}
