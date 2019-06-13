package org.ml4j.nn.synapses;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentChainBaseImpl;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedSynapsesChainImpl<S extends DirectedSynapses<?, ?>> extends DirectedComponentChainBaseImpl<NeuronsActivation, S, DirectedSynapsesActivation, DirectedSynapsesChainActivation> implements DirectedSynapsesChain<S> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedSynapsesChainImpl(
			List<S> components) {
		super(components);
	}

	@Override
	protected DirectedSynapsesChainActivation createChainActivation(
			List<DirectedSynapsesActivation> componentActivations, NeuronsActivation inFlightInput) {
		return new DirectedSynapsesChainActivationImpl(componentActivations, inFlightInput);
	}

	//@Override
	//public DirectedSynapsesChain<L, R> dup() {
	//	return new DirectedSynapsesChainImpl<>(components.stream().map(c -> c.dup()).collect(Collectors.toList()));
	//}

	//@Override
	//protected ChainableDirectedComponentActivation<NeuronsActivation> forwardPropagate(NeuronsActivation input,
	//		DirectedSynapses<L, R> component, int componentIndex,
	//		DirectedComponentsContext context) {
	//	return component.forwardPropagate(input, context.getContext(component));
	//}

	//@Override
	//protected DirectedSynapsesChain<L, R> createDirectedComponentChain(List<DirectedSynapses<L, R>> components) {
///		return new DirectedSynapsesChainImpl<>(components);
	//}

	
}
