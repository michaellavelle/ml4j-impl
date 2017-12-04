package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationWithPossibleBiasUnit;

/**
 * Default implementation of batch-norm DirectedSynapses
 * 
 * @author Michael Lavelle
 *
 * @param <L> The Neurons on the left hand side of these batch-norm DirectedSynapses.
 * @param <R> The Neurons on the right hand side of these batch-norm DirectedSynapses.
 */
public class BatchNormDirectedSynapsesImpl<L extends Neurons, R extends Neurons>
    implements DirectedSynapses<L, R> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;
  private L leftNeurons;
  private R rightNeurons;
  private ScaleAndShiftAxons scaleAndShiftAxons;

  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   * @param scaleAndShiftAxons The scale and shift axons.
   */
  public BatchNormDirectedSynapsesImpl(L leftNeurons, R rightNeurons,
      ScaleAndShiftAxons scaleAndShiftAxons) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.scaleAndShiftAxons = scaleAndShiftAxons;

  }

  @Override
  public DirectedSynapses<L, R> dup() {
    return new BatchNormDirectedSynapsesImpl<L, R>(leftNeurons, rightNeurons,
        scaleAndShiftAxons.dup());
  }

  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesActivation synapsesActivation,
      NeuronsActivation outerGradient, DirectedSynapsesContext context, boolean outer, double reg) {
    if (outer) {
      throw new IllegalStateException("Batch Norm Synapses can't be the outer synapses");
    }

    NeuronsActivationWithPossibleBiasUnit xhatn1 =
        synapsesActivation.getAxonsActivation().getPostDropoutInputWithPossibleBias()
        .withBiasUnit(false, context);
    
    NeuronsActivation xhatn = new NeuronsActivation(xhatn1.getActivations(), 
        xhatn1.getFeatureOrientation());
    
    Matrix xhat = xhatn.getActivations();
    Matrix dout = outerGradient.getActivations().transpose();

    /**
     * . xhat:1000:101COLUMNS_SPAN_FEATURE_SET dout:100:1000ROWS_SPAN_FEATURE_SET
     * 
     * 
     * 
     */


    // System.out.println(
    // "xhat:" + xhat.getRows() + ":" + xhat.getColumns() + xhatn.getFeatureOrientation());
    // Matrix dbeta = outerGradient.
    // System.out.println(
    // "dout:" + dout.getRows() + ":" + dout.getColumns()
    // + outerGradient.getFeatureOrientation());


    Matrix dgamma = xhat.mul(dout).transpose().rowSums().transpose();


    // gamma, xhat, istd = cache
    // N, _ = dout.shape

    // dbeta = np.sum(dout, axis=0)
    // dgamma = np.sum(xhat * dout, axis=0)
    // dx = (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)

    // return dx, dgamma, dbeta

    // System.out.println("dgamma:" + dgamma.getRows() + ":" + dgamma.getColumns());
    // System.out.println("dbeta:" + dbeta.getRows() + ":" + dbeta.getColumns());
    // System.out.println("xhat:" + xhat.getRows() + ":" + xhat.getColumns());


    Matrix dgammab = context.getMatrixFactory().createMatrix(xhat.getRows(), xhat.getColumns());
    for (int i = 0; i < xhat.getRows(); i++) {
      dgammab.putRow(i, dgamma);
    }

    Matrix dbeta = dout.transpose().rowSums().transpose();

    Matrix dbetab = context.getMatrixFactory().createMatrix(xhat.getRows(), xhat.getColumns());
    for (int i = 0; i < xhat.getRows(); i++) {
      dbetab.putRow(i, dbeta);
    }

    int num = xhat.getRows();

    NeuronsActivation input = synapsesActivation.getInput();

    Matrix meanMatrix = getMeanMatrix(input, context.getMatrixFactory());

    Matrix varianceMatrix =
        getVarianceMatrix(input, context.getMatrixFactory(), meanMatrix);

    Matrix istd = context.getMatrixFactory().createMatrix(varianceMatrix.getRows(),
        varianceMatrix.getColumns());

    for (int r = 0; r < varianceMatrix.getRows(); r++) {
      for (int c = 0; c < varianceMatrix.getColumns(); c++) {
        istd.put(r, c, 1d / varianceMatrix.get(r, c));
      }
    }

    Matrix gammaRow = scaleAndShiftAxons.getScaleRowVector();

    Matrix gamma = context.getMatrixFactory().createMatrix(num, gammaRow.getColumns());
    for (int i = 0; i < num; i++) {
      gamma.putRow(i, gammaRow);
    }

    Matrix dx = gamma.mul(istd).div(num).mul(dout.mul(num).sub(xhat.mul(dgammab)).sub(dbetab));

    NeuronsActivation dxn =
        new NeuronsActivation(dx.transpose(), outerGradient.getFeatureOrientation());

    Matrix axonsGradient = context.getMatrixFactory().createMatrix(2, dgamma.getColumns());
    axonsGradient.putRow(0, dgamma);
    axonsGradient.putRow(1, dbeta);

    return new DirectedSynapsesGradientImpl(dxn, axonsGradient.transpose());

  }

  @Override
  public DirectedSynapsesActivation forwardPropagate(DirectedSynapsesInput synapsesInput,
      DirectedSynapsesContext context) {

    NeuronsActivation input = synapsesInput.getInput();

    Matrix meanMatrix = getMeanMatrix(input, context.getMatrixFactory());

    Matrix varianceMatrix =
        getVarianceMatrix(input, context.getMatrixFactory(), meanMatrix);

    Matrix xhat = input.getActivations().sub(meanMatrix).divi(varianceMatrix);

    NeuronsActivation xhatN =
        new NeuronsActivation(xhat, synapsesInput.getInput().getFeatureOrientation());

    // y = gamma * xhat + beta
    AxonsActivation axonsActivation =
        scaleAndShiftAxons.pushLeftToRight(xhatN, null, context.createAxonsContext());

    return new DirectedSynapsesActivationImpl(this, input, axonsActivation,
        axonsActivation.getOutput());
  }

  /**
   * Naive implementation to construct a variance row vector with an entry for each feature.
   * 
   * @param matrix The input matrix
   * @param matrixFactory The matrix factory.
   * @param meanRowVector The mean row vector.
   * @return A row vector the the variances.
   */
  private Matrix getVarianceRowVector(Matrix matrix, MatrixFactory matrixFactory,
      Matrix meanRowVector) {
    Matrix rowVector = matrixFactory.createMatrix(1, matrix.getColumns());
    for (int c = 0; c < matrix.getColumns(); c++) {
      double total = 0d;
      double count = 0;
      for (int r = 0; r < matrix.getRows(); r++) {
        double diff = (matrix.get(r, c) - meanRowVector.get(c));
        total = total + diff * diff;
        count++;
      }
      double variance = total / (count - 1);

      double epsilion = 0.00000001;
      double varianceVal = Math.sqrt(variance * variance + epsilion);
      rowVector.put(0, c, varianceVal);
    }
    return rowVector;
  }

  private Matrix getVarianceMatrix(NeuronsActivation input, MatrixFactory matrixFactory,
      Matrix meanRowVector) {

    Matrix varianceMatrix = matrixFactory.createMatrix(input.getActivations().getRows(),
        input.getActivations().getColumns());
    for (int r = 0; r < varianceMatrix.getRows(); r++) {
      varianceMatrix.putRow(r,
          getVarianceRowVector(input.getActivations(), matrixFactory, meanRowVector));
    }

    return varianceMatrix;
  }

  private Matrix getMeanRowVector(Matrix matrix, MatrixFactory matrixFactory) {
    Matrix rowVector = matrixFactory.createMatrix(1, matrix.getColumns());
    for (int c = 0; c < matrix.getColumns(); c++) {
      double mean = matrix.getColumn(c).sum() / matrix.getRows();
      rowVector.put(0, c, mean);
    }
    return rowVector;
  }

  private Matrix getMeanMatrix(NeuronsActivation input, MatrixFactory matrixFactory) {

    Matrix meanMatrix = matrixFactory.createMatrix(input.getActivations().getRows(),
        input.getActivations().getColumns());
    for (int r = 0; r < meanMatrix.getRows(); r++) {
      meanMatrix.putRow(r, getMeanRowVector(input.getActivations(), matrixFactory));
    }

    return meanMatrix;
  }

  @Override
  public L getLeftNeurons() {
    return leftNeurons;
  }

  @Override
  public R getRightNeurons() {
    return rightNeurons;
  }

  @Override
  public DifferentiableActivationFunction getActivationFunction() {
    return null;
  }

  @Override
  public Axons<?, ?, ?> getAxons() {
    return scaleAndShiftAxons;
  }
}
