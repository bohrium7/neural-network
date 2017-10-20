package neuralNetwork;

import cern.colt.function.DoubleDoubleFunction;
import cern.colt.function.DoubleFunction;

/**
 * Miscellaneous Functions class
 * Contains various formula calculations applied to the neural net
 * 
 * Created by Braden D'Eith and Jason Chan on 2017-03-27.
 * Comp 4932 Assignment 3
 */

public class MiscFunction {

    /**
     * The sigmoid function
     *
     * @param z zeta
     * @return sigmoid
     */
    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Method to add 2 matrices together
     * 
     * @param a first matrix
     * @param b second matrix
     * 
     * @return result of calculation
     */
    public static DoubleDoubleFunction plus = new DoubleDoubleFunction() {
        public double apply(double a, double b) {
            return a+b;
        }
    };
    
    /**
     * Method to multiply 2 matrices together element wise
     * 
     * @param a first matrix
     * @param b second matrix
     * 
     * @return result of calculation
     */
    public static DoubleDoubleFunction mult = new DoubleDoubleFunction() {
        public double apply(double a, double b) {
            return a * b;
        }
    };
    
    /**
     * Method to calculate the cost derivative.
     * 
     * @param a 
     * @param b
     * 
     * @returns the vector of partial derivative, a for the output activations
     */
    public static DoubleDoubleFunction cost_derivative = new DoubleDoubleFunction() {
        public double apply(double a, double b) {
            return a - b;
        }
    };

    //The sigmoid function
    public static DoubleFunction sigmoid = new DoubleFunction() {
        public double apply(double z) {
            return sigmoid(z);
        }
    };

    //Derivative of the sigmoid function
    public static DoubleFunction sigmoidPrime = new DoubleFunction() {
        public double apply(double z) {
            return sigmoid(z) * (1 - sigmoid(z));
        }
    };
}
