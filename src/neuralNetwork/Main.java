package neuralNetwork;

import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;

/**
 * 
 * @author Braden D'Eith & Jason Chan
 * 
 * Comp 4932 Assignment 3 Neural Network
 *
 */
public class Main {
    
    static class NetworkThread implements Runnable {
        int[] net;
        int[][] results;
        int column, epochs, mini_batch_size, value = 0;
        double eta;
        List<Image> training_data, test_data;
        
        public NetworkThread(int[] n, int[][] res, int col, List<Image> training_data, int epochs, int mini_batch_size, double eta, List<Image> test_data) {
            this.net = n;
            this.results = res;
            this.column = col;
            this.training_data = training_data;
            this.epochs = epochs;
            this.mini_batch_size = mini_batch_size;
            this.eta = eta;
            this.test_data = test_data;
        }

        @Override
        public void run() {
            for (int r = 0; r < 7; ++r) {
                Network network = new Network(net);
                results[r][column] = network.SGD(training_data, epochs, mini_batch_size, eta, test_data);
            }
        }
        
        
    }
    
    /**
     * Main entry point for the application 
     * 
     * @param args command line arguments as a string
     */
    public static void main(String[] args) {
        int[] n = {784, 30, 10};
        Network net = new Network(n);
        
        
        Screen screen = new Screen(net);
        screen.init();
        screen.sizeScreen(400, 470);
        screen.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        screen.setVisible(true);
        
        //testNetwork();
    }
    
    /**
     * Function for testing the neural network
     */
    public static void testNetwork() {
        int[] n = {784, 30, 10};
        Network net = new Network(n);
        List<Image> training_data = Network.getImages(true);
        List<Image> test_data = Network.getImages(false);
        
        
        int finalVal = net.SGD(training_data, 1, 50, 3.0, test_data);
        
        for (int i = 0; i < 10; ++i) {
            DoubleMatrix2D number = training_data.get(i).image;//Network.loadImageToMatrix("Test_Images/" + i + ".png");
            
            for (int y = 0; y < 28; ++y){
                for (int x = 0; x < 28; ++x) {
                    System.err.print((int)(number.get(y * 28 + x, 0) * 256) + "\t");
                }
                System.err.println();
           }
            System.err.println(net.feedForward(number));
            System.err.println("Guess: " + Network.getMaxIndex(net.feedForward(number)) +
                    "\nAnswer: " + Network.getMaxIndex(training_data.get(i).label));
        }
            
    }
    
    /**
     * Run extensive tests on different configurations of the neural network to determine the
     * optimal setup.
     */
    public static void runExtensiveTests() {
        List<Image> training_data = Network.getImages(true);
        List<Image> test_data = Network.getImages(false);
        
        List<int[]> networkData    = new ArrayList<>();
        
        //single layer
        networkData.add(new int[]{784, 30, 10});
        networkData.add(new int[]{784, 10, 10});
        
        //double layer
        networkData.add(new int[]{784, 40, 20, 10});
        networkData.add(new int[]{784, 20, 40, 10});
        
        //triple layer
        networkData.add(new int[]{784, 40, 30, 20, 10});
        networkData.add(new int[]{784, 20, 30, 40, 10});
        
        //quad layer
        networkData.add(new int[]{784, 50, 40, 30, 20, 10});
        networkData.add(new int[]{784, 20, 30, 40, 50, 10});
        
        int testNo = 0;
        
        for (int[] n : networkData) { //for each network configuration
            for (int eta = 30; eta <= 50; eta += 20) { //loop on eta
                for (int epochs = 5; epochs <= 10; epochs += 5) { //loop on epochs
                    
                    int[][] results = new int[7][3];
                    Thread[] threads = new Thread[3];
                    
                    for (int batchSize = 10, c = 0; batchSize <= 50; batchSize += 20, ++c) { //loop on batch size
                        threads[c] = new Thread(new NetworkThread(n, results, c, training_data, epochs, batchSize, eta / 10.0, test_data));
                        
                        threads[c].start();
                    }
                    //print out table
                    try {
                        for(int i = 0; i < threads.length; ++i)
                            threads[i].join();
                    } catch (Exception e) {
                    }
                    outputTable(testNo++, eta / 10.0, n.length - 2, n, epochs, results);
                }
            }
        }
        
    }
    
    /**
     * Print the results from the neural network to standard output
     * 
     * @param testNo Test number
     * @param eta Eta of the network (amount of change the network makes while learning)
     * @param layers Number of layers in the network (not including input and output)
     * @param neurons Number of neurons in each layer
     * @param epochs Epochs ran for learning
     * @param results Results from the network
     */
    public static void outputTable(int testNo, double eta, int layers, int[] neurons, int epochs, int[][] results) {
        System.out.println();
        System.out.print("Test #,");
        for (int i = testNo * 3; i < testNo * 3 + 3; i++)
            System.out.print(i + ",");
        System.out.println();
        
        System.out.print("ETA,");
        for (int i = 0; i < 3; i++)
            System.out.print(eta + ",");
        System.out.println();
        
        System.out.print("Layers,");
        for (int i = 0; i < 3; i++)
            System.out.print(layers + ",");
        System.out.println();
        
        for (int i = 1; i < neurons.length - 1; i++) {
            System.out.print(i == 1 ? "Neurons," : ",");
            for (int j = 0; j < 3; j++)
                System.out.print(neurons[i] + ",");
            System.out.println();
        }
            
        System.out.print("Epoch,");
        for (int i = 0; i < 3; ++i)
            System.out.print(epochs + ",");
        System.out.println();
        
        System.out.print("Batch Size,");
        for (int batchSize = 10; batchSize <= 50; batchSize += 20)
            System.out.print(batchSize + ",");
        System.out.println();
        
        for (int r = 0; r < 7; ++r) {
            System.out.print("Result " + r + ",");
            for (int c = 0; c < 3; ++c) {
                System.out.print(results[r][c] + ",");
            }
            System.out.println();
        }
        
        System.out.print("Average,");  
        for (int col = 0; col < 3; ++col) {
            int sum = 0;
            for (int row = 0; row < 7; ++row) {
                sum += results[row][col];
            }
            System.out.print(sum / 7.0 + ",");
        }
        System.out.println();
        
    }
    
    

}
