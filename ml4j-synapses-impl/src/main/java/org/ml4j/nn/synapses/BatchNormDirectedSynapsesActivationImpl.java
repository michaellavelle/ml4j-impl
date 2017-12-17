package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationWithPossibleBiasUnit;

public class BatchNormDirectedSynapsesActivationImpl extends DirectedSynapsesActivationBase {
  
  private ScaleAndShiftAxons scaleAndShiftAxons;
  private Matrix varianceMatrix;
  
  /**
   * @param synapses The synapses.
   * @param scaleAndShiftAxons The scale and shift axons.
   * @param inputActivation The input activation.
   * @param axonsActivation The axons activation.
   * @param activationFunctionActivation The activation function activation.
   * @param outputActivation The output activation.
   */
  public BatchNormDirectedSynapsesActivationImpl(DirectedSynapses<?, ?> synapses, 
      ScaleAndShiftAxons scaleAndShiftAxons, 
      NeuronsActivation inputActivation, AxonsActivation axonsActivation,
      DifferentiableActivationFunctionActivation activationFunctionActivation,
      NeuronsActivation outputActivation, Matrix varianceMatrix) {
    super(synapses, inputActivation, axonsActivation, 
        activationFunctionActivation, outputActivation);
    this.scaleAndShiftAxons = scaleAndShiftAxons;
    this.varianceMatrix = varianceMatrix;
  }

  @Override
  public DirectedSynapsesGradient backPropagate(DirectedSynapsesGradient outerGradient,
      DirectedSynapsesContext context) {
 
    NeuronsActivationWithPossibleBiasUnit xhatn1 =
        getAxonsActivation().getPostDropoutInputWithPossibleBias()
        .withBiasUnit(false, context);
    
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
        new NeuronsActivation(dx.transpose(), outerGradient.getOutput().getFeatureOrientation());

    Matrix axonsGradient = context.getMatrixFactory().createMatrix(2, dgamma.getColumns());
    axonsGradient.putRow(0, dgamma);
    axonsGradient.putRow(1, dbeta);

    return new DirectedSynapsesGradientImpl(dxn, axonsGradient.transpose());
  }
  
  
  @Override
  public DirectedSynapsesGradient backPropagate(CostFunctionGradient outerGradient,
      DirectedSynapsesContext context) {
 
    AxonsContext axonsContext = context.getAxonsContext(0);
    if (axonsContext.getLeftHandInputDropoutKeepProbability() != 1d) {
      throw new UnsupportedOperationException(
          "Reguarlisation of batch norm synapses not yet supported");
    }

    NeuronsActivationWithPossibleBiasUnit xhatn1 =
        getAxonsActivation().getPostDropoutInputWithPossibleBias()
        .withBiasUnit(false, context);
    
    NeuronsActivation xhatn = new NeuronsActivation(xhatn1.getActivations(), 
        xhatn1.getFeatureOrientation());
    
    Matrix xhat = xhatn.getActivations();
    
    Matrix dout = outerGradient.backPropagateThroughFinalActivationFunction(
        activationFunctionActivation
        .getActivationFunction()).getOutput().getActivations().transpose();

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
        new NeuronsActivation(dx.transpose(), outerGradient
            .backPropagateThroughFinalActivationFunction(
            activationFunctionActivation.getActivationFunction()).getOutput()
            .getFeatureOrientation());

    Matrix axonsGradient = context.getMatrixFactory().createMatrix(2, dgamma.getColumns());
    axonsGradient.putRow(0, dgamma);
    axonsGradient.putRow(1, dbeta);
    

    return new DirectedSynapsesGradientImpl(dxn, axonsGradient.transpose());
  }
  
 

  @Override
  public double getTotalRegularisationCost(DirectedSynapsesContext synapsesContext) {
    AxonsContext axonsContext = synapsesContext.getAxonsContext(0);
    if (axonsContext.getLeftHandInputDropoutKeepProbability() != 1d) {
      throw new UnsupportedOperationException(
          "Reguarlisation of batch norm synapses not yet supported");
    }
    return 0;
  }
}
