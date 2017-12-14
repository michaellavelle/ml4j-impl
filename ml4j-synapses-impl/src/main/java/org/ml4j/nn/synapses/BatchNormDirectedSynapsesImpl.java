package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

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
  private DifferentiableActivationFunction activationFunction;

  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   * @param scaleAndShiftAxons The scale and shift axons.
   */
  public BatchNormDirectedSynapsesImpl(L leftNeurons, R rightNeurons,
      ScaleAndShiftAxons scaleAndShiftAxons, DifferentiableActivationFunction activationFunction) {
    this.leftNeurons = leftNeurons;
    this.rightNeurons = rightNeurons;
    this.scaleAndShiftAxons = scaleAndShiftAxons;
    this.activationFunction = activationFunction;

  }

  @Override
  public DirectedSynapses<L, R> dup() {
    return new BatchNormDirectedSynapsesImpl<L, R>(leftNeurons, rightNeurons,
        scaleAndShiftAxons.dup(), activationFunction);
  }

  @Override
  public DirectedSynapsesActivation forwardPropagate(DirectedSynapsesInput synapsesInput,
      DirectedSynapsesContext context) {

    NeuronsActivation input = synapsesInput.getInput();

    Matrix meanMatrix = getMeanMatrix(input, context.getMatrixFactory());

    Matrix varianceMatrix = getVarianceMatrix(input, context.getMatrixFactory(), meanMatrix);

    Matrix xhat = input.getActivations().sub(meanMatrix).divi(varianceMatrix);

    NeuronsActivation xhatN =
        new NeuronsActivation(xhat, synapsesInput.getInput().getFeatureOrientation());

    // y = gamma * xhat + beta
    AxonsActivation axonsActivation =
        scaleAndShiftAxons.pushLeftToRight(xhatN, null, context.createAxonsContext());

    DifferentiableActivationFunctionActivation activationFunctionActivation =
        activationFunction.activate(axonsActivation.getOutput(), context);

    return new BatchNormDirectedSynapsesActivationImpl(this, scaleAndShiftAxons, input,
        axonsActivation, activationFunctionActivation, axonsActivation.getOutput(), varianceMatrix);
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
    return activationFunction;
  }

  @Override
  public Axons<?, ?, ?> getAxons() {
    return scaleAndShiftAxons;
  }
}
