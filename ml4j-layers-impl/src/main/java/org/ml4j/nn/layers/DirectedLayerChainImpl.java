package org.ml4j.nn.layers;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentChainBaseImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedLayerChainImpl<L extends DirectedLayer<?, ?>> extends DirectedComponentChainBaseImpl<NeuronsActivation, L, DirectedLayerActivation, DirectedLayerChainActivation> implements DirectedLayerChain<L> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedLayerChainImpl(
			List<L> components) {
		super(components);
	}

	@Override
	protected DirectedLayerChainActivation createChainActivation(
			List<DirectedLayerActivation> componentActivations, NeuronsActivation inFlightInput) {
		return new DirectedLayerChainActivationImpl(componentActivations);
	}	
}
