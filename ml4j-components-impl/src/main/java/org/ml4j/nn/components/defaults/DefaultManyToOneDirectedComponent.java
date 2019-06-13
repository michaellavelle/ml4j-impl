package org.ml4j.nn.components.defaults;

import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.ManyToOneDirectedComponent;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultManyToOneDirectedComponent extends ManyToOneDirectedComponent<NeuronsActivation, DirectedComponentsContext> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	protected NeuronsActivation getCombinedOutput(List<NeuronsActivation> gradient, DirectedComponentsContext context) {
		
		Matrix totalMatrix = context.getMatrixFactory().createMatrix(gradient.get(0).getActivations().getRows(), gradient.get(0).getActivations().getColumns());
		for (NeuronsActivation activation : gradient) {
			totalMatrix.add(activation.getActivations());
		}
		return new NeuronsActivation(totalMatrix,  gradient.get(0).getFeatureOrientation());
	}

}
