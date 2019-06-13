package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationWithPossibleBiasUnit;

public class BatchNormDirectedAxonsComponentActivationImpl implements DirectedAxonsComponentActivation {
  
  private ScaleAndShiftAxons scaleAndShiftAxons;
  private AxonsActivation scaleAndShiftAxonsActivation;
  private Matrix meanMatrix;
  private Matrix varianceMatrix;
  private BatchNormDirectedAxonsComponent<?, ?> batchNormAxons;
  private AxonsContext axonsContext;
  
  /**
   * @param synapses The synapses.
   * @param scaleAndShiftAxons The scale and shift axons.
   * @param inputActivation The input activation.
   * @param axonsActivation The axons activation.
   * @param activationFunctionActivation The activation function activation.
   * @param outputActivation The output activation.
   */
  public BatchNormDirectedAxonsComponentActivationImpl(BatchNormDirectedAxonsComponent<?, ?> batchNormAxons,
      ScaleAndShiftAxons scaleAndShiftAxons, 
      AxonsActivation scaleAndShiftAxonsActivation,
      Matrix meanMatrix, Matrix varianceMatrix, AxonsContext axonsContext) {
	  this.batchNormAxons = batchNormAxons;
    this.scaleAndShiftAxons = scaleAndShiftAxons;
    this.scaleAndShiftAxonsActivation = scaleAndShiftAxonsActivation;
    //this.primaryAxonsActivation = primaryAxonsActivation;
    //this.primaryAxons = primaryAxons;
    this.meanMatrix = meanMatrix;
    this.varianceMatrix = varianceMatrix;
  }

  @Override
  public DirectedComponentGradient<NeuronsActivation> backPropagate(DirectedComponentGradient<NeuronsActivation> outerGradient) {
    
    // Build up the exponentially weighted averages
    
    Matrix exponentiallyWeightedAverageMean = 
    		batchNormAxons.getExponentiallyWeightedAverageInputFeatureMeans();
    
    double beta = batchNormAxons.getBetaForExponentiallyWeightedAverages();
    
    if (exponentiallyWeightedAverageMean == null) {
      exponentiallyWeightedAverageMean = meanMatrix.getRow(0);
    } else {
      exponentiallyWeightedAverageMean =
          exponentiallyWeightedAverageMean.mul(beta).add(meanMatrix.getRow(0).mul(1 - beta));
    }
    batchNormAxons
        .setExponentiallyWeightedAverageInputFeatureMeans(exponentiallyWeightedAverageMean);

    Matrix exponentiallyWeightedAverageVariance =
    		batchNormAxons.getExponentiallyWeightedAverageInputFeatureVariances();


    if (exponentiallyWeightedAverageVariance == null) {
      exponentiallyWeightedAverageVariance = varianceMatrix.getRow(0);
    } else {
      exponentiallyWeightedAverageVariance = exponentiallyWeightedAverageVariance.mul(beta)
          .add(varianceMatrix.getRow(0).mul(1 - beta));
    }
    batchNormAxons
        .setExponentiallyWeightedAverageInputFeatureVariances(exponentiallyWeightedAverageVariance);

 
    NeuronsActivationWithPossibleBiasUnit xhatn1 =
    		scaleAndShiftAxonsActivation.getPostDropoutInputWithPossibleBias()
        .withBiasUnit(false, axonsContext);
    
    NeuronsActivation xhatn = new NeuronsActivation(xhatn1.getActivations(), 
        xhatn1.getFeatureOrientation());
    
    Matrix xhat = xhatn.getActivations();
    
    Matrix dout = outerGradient.getOutput().getActivations().transpose();
    
    

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


    Matrix dgammab = axonsContext.getMatrixFactory().createMatrix(xhat.getRows(), xhat.getColumns());
    for (int i = 0; i < xhat.getRows(); i++) {
      dgammab.putRow(i, dgamma);
    }

    Matrix dbeta = dout.transpose().rowSums().transpose();

    Matrix dbetab = axonsContext.getMatrixFactory().createMatrix(xhat.getRows(), xhat.getColumns());
    for (int i = 0; i < xhat.getRows(); i++) {
      dbetab.putRow(i, dbeta);
    }

    int num = xhat.getRows();

    Matrix istd = axonsContext.getMatrixFactory().createMatrix(varianceMatrix.getRows(),
        varianceMatrix.getColumns());

    for (int r = 0; r < varianceMatrix.getRows(); r++) {
      for (int c = 0; c < varianceMatrix.getColumns(); c++) {
        istd.put(r, c, 1d / varianceMatrix.get(r, c));
      }
    }

    Matrix gammaRow = scaleAndShiftAxons.getScaleRowVector();

    Matrix gamma = axonsContext.getMatrixFactory().createMatrix(num, gammaRow.getColumns());
    for (int i = 0; i < num; i++) {
      gamma.putRow(i, gammaRow);
    }

    Matrix dx = gamma.mul(istd).div(num).mul(dout.mul(num).sub(xhat.mul(dgammab)).sub(dbetab));

    NeuronsActivation dxn =
        new NeuronsActivation(dx.transpose(), outerGradient.getOutput().getFeatureOrientation());

    Matrix axonsGradient = axonsContext.getMatrixFactory().createMatrix(2, dgamma.getColumns());
    axonsGradient.putRow(0, dgamma);
    axonsGradient.putRow(1, dbeta);
    
   
    return new DirectedComponentGradientImpl<>( outerGradient.getTotalTrainableAxonsGradients(),
        new AxonsGradientImpl(scaleAndShiftAxons, axonsGradient.transpose()), dxn);
  }
  
  
 

  @Override
  public double getTotalRegularisationCost(AxonsContext axonsContext ) {
    if (axonsContext.getLeftHandInputDropoutKeepProbability() != 1d) {
      throw new UnsupportedOperationException(
          "Reguarlisation of batch norm synapses not yet supported");
    }
    return 0;
  }

@Override
public NeuronsActivation getOutput() {
	return scaleAndShiftAxonsActivation.getOutput();
}

@Override
public DirectedAxonsComponent<?, ?> getAxonsComponent() {
	return batchNormAxons;
}

@Override
public double getAverageRegularisationCost(AxonsContext axonsContext) {
	 if (axonsContext.getLeftHandInputDropoutKeepProbability() != 1d) {
	      throw new UnsupportedOperationException(
	          "Reguarlisation of batch norm synapses not yet supported");
	    }
	    return 0;
}
}
