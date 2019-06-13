package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultOneToManyDirectedComponentActivation extends OneToManyDirectedComponentActivation<NeuronsActivation> {

	private MatrixFactory matrixFactory;
	
	public DefaultOneToManyDirectedComponentActivation(List<NeuronsActivation> activations, MatrixFactory matrixFactory) {
		super(activations);
		this.matrixFactory = matrixFactory;
	}

	@Override
	protected NeuronsActivation getBackPropagatedGradient(List<NeuronsActivation> gradient) {
		
		Matrix totalMatrix = matrixFactory.createMatrix(gradient.get(0).getActivations().getRows(), gradient.get(0).getActivations().getColumns());
		for (NeuronsActivation activation : gradient) {
			totalMatrix.add(activation.getActivations());
		}
		return new NeuronsActivation(totalMatrix,  gradient.get(0).getFeatureOrientation());
	}

}
