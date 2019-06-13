package org.ml4j.nn.axons;

import java.util.Arrays;
import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
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
public class BatchNormDirectedAxonsComponentImpl
    implements BatchNormDirectedAxonsComponent<Neurons, Neurons> {

  /**
   * Default serialization id.
   */
  private static final long serialVersionUID = 1L;

  //private Axons<L, R, ?> primaryAxons;
  private ScaleAndShiftAxons scaleAndShiftAxons;
  
  private Matrix exponentiallyWeightedAverageInputFeatureMeans;
  private Matrix exponentiallyWeightedAverageInputFeatureVariances;
  private double betaForExponentiallyWeightedAverages;

  /**
   * @param leftNeurons The left neurons.
   * @param rightNeurons The right neurons.
   * @param scaleAndShiftAxons The scale and shift axons.
   */
  public BatchNormDirectedAxonsComponentImpl(
      ScaleAndShiftAxons scaleAndShiftAxons) {
    this.scaleAndShiftAxons = scaleAndShiftAxons;
    this.betaForExponentiallyWeightedAverages = 0.9;
    //this.primaryAxons = primaryAxons;
  }



  @Override
  public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation input,
      AxonsContext axonsContext) {
	 
    Matrix meanMatrix = getMeanMatrix(input, axonsContext.getMatrixFactory());

    Matrix varianceMatrix = getVarianceMatrix(input, axonsContext.getMatrixFactory(), meanMatrix);

    Matrix xhat = input.getActivations().sub(meanMatrix).divi(varianceMatrix);

    NeuronsActivation xhatN =
        new NeuronsActivation(xhat, input.getFeatureOrientation());

    // y = gamma * xhat + beta
    AxonsActivation axonsActivation =
    		// TODO ML
        scaleAndShiftAxons.pushLeftToRight(xhatN, null, axonsContext);



    return new BatchNormDirectedAxonsComponentActivationImpl(this, scaleAndShiftAxons, axonsActivation, 
        meanMatrix, varianceMatrix, axonsContext);
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

    if (input.getActivations().getRows() == 1) {
      if (exponentiallyWeightedAverageInputFeatureVariances != null) {
        return exponentiallyWeightedAverageInputFeatureVariances;
      } else {
        throw new IllegalStateException("Unable to calcuate mean and variance for batch "
            + "norm on a single example - no exponentially weighted average available");
      }
    } else {

      Matrix varianceMatrix = matrixFactory.createMatrix(input.getActivations().getRows(),
          input.getActivations().getColumns());
      for (int r = 0; r < varianceMatrix.getRows(); r++) {
        varianceMatrix.putRow(r,
            getVarianceRowVector(input.getActivations(), matrixFactory, meanRowVector));
      }

      return varianceMatrix;
    }
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

    if (input.getActivations().getRows() == 1) {
      if (exponentiallyWeightedAverageInputFeatureMeans != null) {
        return exponentiallyWeightedAverageInputFeatureMeans;
      } else {
        throw new IllegalStateException("Unable to calcuate mean and variance for batch "
            + "norm on a single example - no exponentially weighted average available");
      }
    } else {
      Matrix meanMatrix = matrixFactory.createMatrix(input.getActivations().getRows(),
          input.getActivations().getColumns());
      for (int r = 0; r < meanMatrix.getRows(); r++) {
        meanMatrix.putRow(r, getMeanRowVector(input.getActivations(), matrixFactory));
      }
      return meanMatrix;
    }
  }

 
  /*
  @Override
  public DirectedDipoleGraph<Axons<?, ?, ?>> getAxonsGraph() {
    return new DirectedDipoleGraphImpl<Axons<?, ?, ?>>(scaleAndShiftAxons);
  }

  @Override
  public Axons<?, ?, ?> getPrimaryAxons() {
    return scaleAndShiftAxons;
  }
*/
  @Override
  public double getBetaForExponentiallyWeightedAverages() {
    return betaForExponentiallyWeightedAverages;
  }

  @Override
  public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
    return exponentiallyWeightedAverageInputFeatureMeans;
  }

  @Override
  public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
    return exponentiallyWeightedAverageInputFeatureVariances;
  }

  @Override
  public void setExponentiallyWeightedAverageInputFeatureMeans(
      Matrix exponentiallyWeightedAverageInputFeatureMeans) {
    this.exponentiallyWeightedAverageInputFeatureMeans =
        exponentiallyWeightedAverageInputFeatureMeans;
  }

  @Override
  public void setExponentiallyWeightedAverageInputFeatureVariances(
      Matrix exponentiallyWeightedAverageInputFeatureVariances) {
    this.exponentiallyWeightedAverageInputFeatureVariances =
        exponentiallyWeightedAverageInputFeatureVariances;
  }

  @Override
  public AxonsContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
	return directedComponentsContext.getContext(this, () -> new AxonsContextImpl(directedComponentsContext.getMatrixFactory(), false));
  }

@Override
public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
	return Arrays.asList(new DirectedAxonsComponentImpl<>(scaleAndShiftAxons));
}



@Override
public Axons<Neurons, Neurons, ?> getAxons() {
	return scaleAndShiftAxons;
}


}
