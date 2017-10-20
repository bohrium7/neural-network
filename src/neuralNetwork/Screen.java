package neuralNetwork;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import java.util.List;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;

class DrawListender extends MouseAdapter {
    Pixel pixel;
    Screen screen;
    
    /**
     * Constructor.
     * 
     * @param pi The pixel that this listener is listening to.
     * */
    public DrawListender(Pixel pi, Screen scr) {
        pixel = pi;
        screen = scr;
    }
    
    /**
     * Triggers when the mouse is dragged.  draws a point on the pixel.
     * 
     * @param e The mouse Event.
     * */
    public void mouseEntered(MouseEvent ent) {
        if (SwingUtilities.isLeftMouseButton(ent))
            pixel.setIntensity(1);
        else if (SwingUtilities.isRightMouseButton(ent))
            pixel.setIntensity(0);
    }
}

public class Screen extends JFrame{
    static final int WIDTH  = 28,
                     HEIGHT = 28;
    Pixel[][] pixels = new Pixel[WIDTH][HEIGHT]; 
    Network network;
    
    public Screen(Network n) {
        network = n;
    }
    
    /**
     * Initialize the screen
     */
    public void init()
    {
        setTitle("Draw a number!");
        setLayout(new FlowLayout());
        
        add(setupButtons());
        
        JPanel drawing = new JPanel();
        
        drawing.setLayout(new GridLayout(28, 28));

        for(int row = 0; row < HEIGHT; row++)
        {
            for(int col = 0; col < WIDTH; col++)
            {
                pixels[row][col] = new Pixel();
                pixels[row][col].addMouseListener(new DrawListender(pixels[row][col], this));
                drawing.add(pixels[row][col]);
            }
        }
        
        add(drawing);
    }
    
    /**
     * sets up the buttons used to interact with the neural network
     * 
     * @return a JPanel containing the enter, reset and learn buttons
     */
    private JPanel setupButtons() {
        //setup feedforward button
        JPanel buttons = new JPanel();
        
        //Setup the enter button
        JButton enterButton = new JButton();
        enterButton.setText("Enter");
        enterButton.addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent ent) {
                blur();
                
                DoubleMatrix2D number = getImage();
                DoubleMatrix2D result = network.feedForward(number);
                
                for (int y = 0; y < 28; ++y){
                    for (int x = 0; x < 28; ++x) {
                        System.err.print((int)(number.get(y * 28 + x, 0) * 256) + "\t");
                    }
                    System.err.println();
               }
                
               System.err.println(result);
               System.err.println("Guess: " + Network.getMaxIndex(result));    
               
           	JOptionPane.showMessageDialog(null, "My guess is you wrote a\n" + Network.getMaxIndex(result), "Neural Network", 1);

           	reset();
            }            
        });
        
        //Setup reset button
        JButton resetButton = new JButton();
        resetButton.setText("Reset");
        resetButton.addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent ent) {
                reset();           
            }
        });
        
        //setup learn button
        JButton learnButton = new JButton();
        learnButton.setText("Learn");
        learnButton.addMouseListener(new MouseAdapter() {
            public void mouseClicked(MouseEvent ent) {
                JFrame jf = new JFrame("Input Dialog");           	
            	String userInput = JOptionPane.showInputDialog(jf, "Please enter number of epoch");
            	
            	int input = Integer.parseInt(userInput);
            	
                List<Image> training_data = Network.getImages(true);
                List<Image> test_data = Network.getImages(false); 
                
                double result = network.SGD(training_data, input, 10, 3.0, test_data);
                
                JOptionPane.showMessageDialog(null, "I have finished learning!\nI am at " + String.format("%.2f", result / 100.0) + "% accuracy now.", "Neural Network", 1);

                enterButton.setEnabled(true);
                resetButton.setEnabled(true);                
            }
        });
        
        buttons.add(enterButton);
        buttons.add(learnButton);
        buttons.add(resetButton);
        
        enterButton.setEnabled(false);
        resetButton.setEnabled(false);
        
        return buttons;
    }
    
    /**
     * sets the screen size of the screen
     * 
     * @param width width of the screen in pixels
     * @param height height of the screen in pixels
     */
    public void sizeScreen(int width, int height) {
        final Dimension size;

        size = new Dimension(width, height);
        setSize(size);
        
    }
    
    /**
     * Reset the pixel value in the screen to black
     */
    public void reset() {
        for(int row = 0; row < HEIGHT; row++)
            for(int col = 0; col < WIDTH; col++)
                pixels[row][col].setIntensity(0);
    }
    
    /**
     * Blur the pixel values of the screen
     */
    public void blur() {
        double[][] kernel = {
                {1.0/9.0, 1.0/9.0, 1.0/9.0},
                {1.0/9.0, 1.0/9.0, 1.0/9.0},
                {1.0/9.0, 1.0/9.0, 1.0/9.0}
        };
        
        for(int row = 1; row < HEIGHT - 1; row++) {
            for(int col = 1; col < WIDTH - 1; col++) {
                pixels[row][col].setIntensity(applyKernel(kernel, row, col));
            }
        }
    }
    
    /**
     * Apply the kernel to the screen's pixel values
     * 
     * @param a Kernel to apply
     * @param row Source row
     * @param col Source column
     * @return sum of the applied kernel
     */
    public double applyKernel(double[][] a, int row, int col) {
        double sum = 0;
        
        for (int r = -1; r <= 1; ++r ) {
            for (int c = -1; c <= 1; ++c) {
                sum += a[r+1][c+1] * pixels[row + r][col + c].getIntensity();
            }
        }
        
        return sum;
    }
    
    /**
     * Convert the pixels in the screen to an image object
     * 
     * @return 2D matrix of doubles representing the image drawn on the screen
     */
    public DoubleMatrix2D getImage() {
        DoubleMatrix2D image = DoubleFactory2D.dense.make(28 * 28, 1);
        int i = 0;
        for (int row = 0; row < 28; row++) {
            for (int column = 0; column < 28; column++) {
                image.set(i++, 0, pixels[row][column].getIntensity());
            }
        }
        
        return image;
    }
}
