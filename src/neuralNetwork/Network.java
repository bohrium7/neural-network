package neuralNetwork;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/**
 * Network class
 * Main engine of this application
 * 
 * Created by Braden D'Eith and Jason Chan on 2017-03-27.
 * Comp 4932 Assignment 3
 */


public class Network {
    private int numLayers;
    private List<DoubleMatrix2D> biases = new ArrayList<>(),
                                 weights = new ArrayList<>();
    private int[] sizes;
    private Algebra algebra = Algebra.DEFAULT;
    private static DoubleFactory2D fact = DoubleFactory2D.dense;
    
    public double result;

    /**
     * Constructor.
     * Initializes the biases and weights randomly based on a gaussian distribution with mean 0 and
     * standard deviation of 1.
     *
     * @param s defines the number of layers to the network and how many neurons are in each
     *          layer. for example [2, 3, 1] would have 2 input neurons, 3 neurons in the hidden
     *          layer and 1 neuron for the output.
     */
    Network(int[] s) {
        //save the sizes array for later use
        sizes = s;

        //saves the number of layers in the neural net
        numLayers = sizes.length;

        //initialize all bias values to gaussian random values
        for (int i = 1; i < numLayers; i++)
            biases.add(fact.make(getGaussianRandom(sizes[i], 1)));

        //initialize the weights to gaussian random values
        for (int x = 0, y = 1; y < numLayers; x++, y++)
            weights.add(fact.make(getGaussianRandom(sizes[y], sizes[x])));
    }

    /**
     * returns the output of the network if a is the input
     *
     * @param a The input into the network
     * @return The output of the network
     */
    public DoubleMatrix2D feedForward(DoubleMatrix2D a) {
        DoubleMatrix2D b, w;

        for (int i = 0; i < biases.size(); i++) {
            b = biases.get(i); //the current bias
            w = weights.get(i); //the current weight
            //a = sigmoid(np.dot(w, a) + b)
            a = algebra.mult(w, a).assign(b, MiscFunction.plus).assign(MiscFunction.sigmoid);
        }
        return a;
    }
    
    /**
     * Train the neural network using mini-batch stochastic gradient descent.  If training data is provided,
     * then the network will be evaluated against the test data after each epoch, and partial progress
     * printed out.
     * 
     * @param training_data List of tuples "x, y" representing the training inputs and the desired outputs
     * @param epochs The number of epochs to run through
     * @param mini_batch_size the size of the epochs
     * @param eta the amount of change per epoch
     * @param test_data the test data to verify how well the epoch of training did
     */
    public int SGD(List<Image> training_data, int epochs, int mini_batch_size, double eta, List<Image> test_data) {
        int n_test = 0, finalValue = 0;
        
        if (test_data != null)
            n_test = test_data.size();
        int n = training_data.size();
        
        for (int j = 0; j < epochs; ++j) {
            Collections.shuffle(training_data);
            for (int k = 0; k < n; k += mini_batch_size) {
                update_mini_batch(training_data.subList(k, k + mini_batch_size), eta);
            }
            /*
            if (test_data != null) {
               // System.err.println("Epoch " + j + ": " + (finalValue = evaluate(test_data)) + " / " + n_test);

            } else {
                //System.err.println("Epoch " + j + " complete.");
            }
            */
        }
        return evaluate(test_data);
    }
    
    /**
     * Updates the network's weights and biases by applying gradient descent using back propagation
     * to a single mini_batch.
     * 
     * @param mini_batch List of tuples "x, y"
     * @param eta The learning rate
     */
    public void update_mini_batch(List<Image> mini_batch, double eta) {
        List<DoubleMatrix2D> nabla_b = new ArrayList<>(),
                             nabla_w = new ArrayList<>();
        
        //initialize nabla_b and nabla_w to be same sizes as biases and weights
        for (DoubleMatrix2D b : biases)
            nabla_b.add(fact.make(b.rows(), b.columns()));
        for (DoubleMatrix2D w : weights)
            nabla_w.add(fact.make(w.rows(), w.columns()));
        
        for (Image image : mini_batch) {
            //delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            MatrixPair result = backprop(image);
            List<DoubleMatrix2D> delta_nabla_b = result.nabla_b,
                                 delta_nabla_w = result.nabla_w;
            
            //nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            for (int i = 0; i < nabla_b.size(); i++)
                nabla_b.get(i).assign(delta_nabla_b.get(i), MiscFunction.plus);
            
            //nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            for (int i = 0; i < nabla_w.size(); i++)
                nabla_w.get(i).assign(delta_nabla_w.get(i), MiscFunction.plus); 
        }
        
        for (int i = 0; i < weights.size(); i++) {
            DoubleMatrix2D w = weights.get(i), nw = nabla_w.get(i),
                    etaDSize = fact.make(w.rows(), w.columns(), eta / (double)mini_batch.size());
            nw.assign(etaDSize, MiscFunction.mult);
            
            weights.set(i, w.assign(nw, MiscFunction.cost_derivative));
        }
        for (int i = 0; i < biases.size(); i++) {
            DoubleMatrix2D b = biases.get(i), nb = nabla_b.get(i),
                    etaDSize = fact.make(b.rows(), b.columns(), eta / (double)mini_batch.size());
            nb.assign(etaDSize, MiscFunction.mult);
            
            biases.set(i, b.assign(nb, MiscFunction.cost_derivative));
        }
    }
    
    /**
     * Send in a test image (2D matrix of pixel values and a correct number label) to teach the 
     * neural network what to look for in each number
     * 
     * @param Input group of 2D matrices representing the correct number and its pixel values
     * @return Change in weights and biases used for the machine learning
     */
    public MatrixPair backprop(Image input){
        
        List<DoubleMatrix2D> nabla_b = new ArrayList<>(),
                             nabla_w = new ArrayList<>();

        for(DoubleMatrix2D b : biases){
            nabla_b.add(fact.make(b.rows(), b.columns()));
        }
        for(DoubleMatrix2D w : weights){
            nabla_w.add(fact.make(w.rows(), w.columns()));
        }
        
        //feed forward
        DoubleMatrix2D activation = input.image;
        List<DoubleMatrix2D> activations = new ArrayList<>();
        activations.add(activation.copy());
        
        DoubleMatrix2D z;
        List<DoubleMatrix2D> zs = new ArrayList<>();
        
        DoubleMatrix2D b, w;
        
        for (int i = 0; i < biases.size(); i++ ){
            b = biases.get(i);
            w = weights.get(i);
            
            z = algebra.mult(w, activation).assign(b, MiscFunction.plus);
            zs.add(z.copy());
            
            activation = z.assign(MiscFunction.sigmoid).copy();
            activations.add(activation.copy());
        }
        
        // backward pass
        int aSize = activations.size();
        int zSize = zs.size();
        int wSize = weights.size();
        
        // delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        DoubleMatrix2D delta = activations.get(aSize - 1).copy().assign(input.label, MiscFunction.cost_derivative).assign(zs.get(zSize - 1).copy().assign(MiscFunction.sigmoidPrime), MiscFunction.mult);
        
        //nabla_b[-1] = delta
        nabla_b.set(nabla_b.size() - 1, delta.copy());
        
        //nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_w.set(nabla_w.size() - 1, algebra.mult(delta, activations.get(aSize - 2).viewDice()));

        // Note that the variable l in the loop below is used a little
        // differently to the notation in Chapter 2 of the book.  Here,
        // l = 1 means the last layer of neurons, l = 2 is the
        // second-last layer, and so on.  It's a renumbering of the
        // scheme in the book, used here to take advantage of the fact
        // that Python can use negative indices in lists.
        
        for(int last = 2; last < numLayers; last++){
            z = zs.get( zSize - last ).copy();
            DoubleMatrix2D sp = z.assign(MiscFunction.sigmoidPrime);
            delta = algebra.mult(weights.get(wSize - last + 1).viewDice(), delta).assign(sp, MiscFunction.mult);

            nabla_b.set(nabla_b.size() - last, delta.copy());
            nabla_w.set(nabla_w.size() - last, algebra.mult(delta, activations.get(aSize - last - 1).viewDice()));
        }
        
        MatrixPair mp = new MatrixPair(nabla_b, nabla_w);
        return mp;
    }
    
    
    /**
     * return the number of test inputs for which the neural network outputs the correct result.
     * Note that the neural network's output is assumed to be the index of whichever neuron in
     * the final layer has the highest activation.
     * 
     * @param test_data the test data to check the neural network with
     * @return the number of numbers that were read correctly
     */
    public int evaluate(List<Image> test_data) {
        int count = 0;
        
        for (Image test : test_data) {
            DoubleMatrix2D test_results = feedForward(test.image);
            
            if (getMaxIndex(test_results) == getMaxIndex(test.label))
                count++;
        }
        
        return count;
    }
    
    /**
     * Searches a two dimensional matrix and returns the max index
     * 
     * @param values 2D matrix of values
     * @return Index which holds the highest value
     */
    public static int getMaxIndex(DoubleMatrix2D values) {
        int maxIndex = -1;
        double maxTest = -1;
        
        for (int y = 0; y < values.rows(); y++) {
            if (values.get(y, 0) > maxTest){
                maxTest = values.get(y, 0);
                maxIndex = y;
            }
        }
        
        return maxIndex;
    }

    /**
     * Creates a matrix of specified width and height with values determined by gaussian
     * distribution with mean 0 and standard deviation of 1.
     *
     * @param width the width of the matrix
     * @param height the height of the matix
     * @return the 2D array of gaussian doubles
     */
    public static double[][] getGaussianRandom(int width, int height) {
        double[][] values = new double[width][height];
        Random rand = new Random();

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                values[x][y] = rand.nextGaussian();
        return values;
    }
    
    /**
     *  Returns either the training data as pairs of Image matrix and label matrix
     *  
     * @param training if getting the training data, or the test data
     * @return the formatted List of images
     */
    public static List<Image> getImages(boolean training) {
        List<DoubleMatrix2D> images = Network.getMnistImages(training ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte"),
                             labels = Network.getMnistLabels(training ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte");
        List<Image> newImages = new ArrayList<>();
        for (int i = 0; i < labels.size(); i++)
            newImages.add(new Image(images.get(i), labels.get(i)));
        
        return newImages;
    }
    
    
    /**
     * gets the images as values between 0 and 1 from the MNIST files
     * @return array of values varying from 0 to 1, representing pixel intensity
     */
    private static List<DoubleMatrix2D> getMnistImages(String path) {
        List<int[][]> images = MnistReader.getImages(path);
        List<DoubleMatrix2D> newImages = new ArrayList<>();
        
        for (int[][] image : images){
            DoubleMatrix2D values = fact.make(28 * 28, 1);
            int i = 0;
            for (int row = 0; row < 28; row++) {
                for (int column = 0; column < 28; column++) {
                    values.set(i++, 0, image[row][column] / 256.0);
                }
            }
            newImages.add(values);
        }
        
        return newImages;
    }
    
    /**
     * Reads all of the number labels from the file specified in the path.  These labels represent
     * the correct numbers to guess from the MNIST learning set
     * 
     * @param path Path to the file containing the labels for the learning set
     * @return array of values which represents the correct number for each learning number (the
     *         index that has a value of 1.0 is the correct number)
     */
    private static List<DoubleMatrix2D> getMnistLabels(String path) {
        int[] labels = MnistReader.getLabels(path);
        List<DoubleMatrix2D> newLabels = new ArrayList<>();
        
        for (int i = 0; i < labels.length; i++) {
            DoubleMatrix2D values = fact.make(10, 1);
            
            values.set(labels[i], 0, 1.0);
            
            newLabels.add(values);
        }
        
        return newLabels;
    }
    
    /**
     * Formats an image to fit into an 28x28 test data, formatted to be a 784x1 matrix
     * 
     * @param path path to a valid image location
     * @return the new matrix for the image
     */
    public static DoubleMatrix2D loadImageToMatrix(String path) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        img.getScaledInstance(28, 28, BufferedImage.SCALE_DEFAULT);
        
        DoubleMatrix2D values = fact.make(img.getWidth() * img.getHeight(), 1);
        int i = 0;
        for (int row = 0; row < 28; row++) {
            for (int column = 0; column < 28; column++) {
                values.set(i++, 0, (img.getRGB(column, row) & 0x000000FF) / 256.0);
            }
        }
        
        return values;
    }
    
    public void writeNetworkToFile(String path) {
        
    }
}
