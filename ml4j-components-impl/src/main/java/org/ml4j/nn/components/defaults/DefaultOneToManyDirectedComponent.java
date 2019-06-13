package org.ml4j.nn.components.defaults;

import java.util.List;
import java.util.function.IntSupplier;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.OneToManyDirectedComponent;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultOneToManyDirectedComponent extends OneToManyDirectedComponent<NeuronsActivation, DirectedComponentsContext> {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;

	public DefaultOneToManyDirectedComponent(IntSupplier outputCount) {
		super(outputCount);
	}

	@Override
	protected OneToManyDirectedComponentActivation<NeuronsActivation> createActivation(List<NeuronsActivation> acts,
			DirectedComponentsContext synapsesContext) {
		return new DefaultOneToManyDirectedComponentActivation(acts, synapsesContext.getMatrixFactory());
	}
}
