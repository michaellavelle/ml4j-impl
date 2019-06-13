package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DirectedAxonsComponentActivationImpl implements DirectedAxonsComponentActivation {
	
	  private static final Logger LOGGER =
		      LoggerFactory.getLogger(DirectedAxonsComponentActivationImpl.class);
	
	private Axons<?, ?, ?> axons;
	private AxonsActivation axonsActivation;
	private AxonsContext axonsContext;
	
	public DirectedAxonsComponentActivationImpl(AxonsActivation axonsActivation, AxonsContext axonsContext) {
		this.axonsActivation = axonsActivation;
		this.axons = axonsActivation.getAxons();
		this.axonsContext = axonsContext;
	}
	

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(DirectedComponentGradient<NeuronsActivation> outerGradient) {
		return backPropagateThroughAxons(outerGradient, axonsContext);
	}

	@Override
	public NeuronsActivation getOutput() {
		return axonsActivation.getOutput();
	}
	
	private DirectedComponentGradient<NeuronsActivation> backPropagateThroughAxons(DirectedComponentGradient<NeuronsActivation> dz,
		      AxonsContext axonsContext) {

		    LOGGER.debug("Pushing data right to left through axons...");
		    
		    // Will contain bias unit if Axons have left bias unit
		    NeuronsActivation inputGradient =
		        axons.pushRightToLeft(dz.getOutput(), axonsActivation, axonsContext).getOutput();

		    Matrix totalTrainableAxonsGradientMatrix = null;
		    AxonsGradient totalTrainableAxonsGradient = null;

		    if (axons instanceof TrainableAxons<?, ?, ?> && axons.isTrainable(axonsContext)) {

		      LOGGER.debug("Calculating Axons Gradients");

		      totalTrainableAxonsGradientMatrix = dz.getOutput().getActivations()
		          .mmul(axonsActivation.getPostDropoutInputWithPossibleBias().getActivationsWithBias());

		      if (axonsContext.getRegularisationLambda() != 0) {

		        LOGGER.debug("Calculating total regularisation Gradients");

		        Matrix connectionWeightsCopy = axons.getDetachedConnectionWeights();

		        Matrix firstRow = totalTrainableAxonsGradientMatrix.getRow(0);
		        Matrix firstColumn = totalTrainableAxonsGradientMatrix.getColumn(0);

		        totalTrainableAxonsGradientMatrix =
		            totalTrainableAxonsGradientMatrix.addi(connectionWeightsCopy.muli(
		                axonsContext.getRegularisationLambda()));

		        if (axons.getLeftNeurons().hasBiasUnit()) {

		          totalTrainableAxonsGradientMatrix.putRow(0, firstRow);
		        }
		        if (axons.getRightNeurons().hasBiasUnit()) {

		          totalTrainableAxonsGradientMatrix.putColumn(0, firstColumn);
		        }
		      }
		      totalTrainableAxonsGradient = new AxonsGradientImpl((TrainableAxons<?, ?, ?>) axons, 
		          totalTrainableAxonsGradientMatrix);
		    }
		    if (totalTrainableAxonsGradient != null) {
		    	return new DirectedComponentGradientImpl<>(dz.getTotalTrainableAxonsGradients(), 
		            totalTrainableAxonsGradient, inputGradient);
		    } else {
		    	return new DirectedComponentGradientImpl<>(dz.getTotalTrainableAxonsGradients(), inputGradient);
		    }
	}


	@Override
	public DirectedAxonsComponent<?, ?> getAxonsComponent() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public double getTotalRegularisationCost(AxonsContext axonsContext) {
		// TODO Auto-generated method stub
		return 0;
	}


	@Override
	public double getAverageRegularisationCost(AxonsContext axonsContext) {
		// TODO Auto-generated method stub
		return 0;
	}

}
