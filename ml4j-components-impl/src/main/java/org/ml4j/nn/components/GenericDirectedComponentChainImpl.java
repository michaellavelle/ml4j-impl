package org.ml4j.nn.components;

import java.util.List;

public abstract class GenericDirectedComponentChainImpl<I, A extends ChainableDirectedComponentActivation<I>> extends DirectedComponentChainBaseImpl<I, ChainableDirectedComponent<I, ? extends A, ?> ,A, DirectedComponentChainActivation<I, A>> implements GenericDirectedComponentChain<I, A> {

	@Override
	protected DirectedComponentChainActivation<I, A> createChainActivation(List<A> componentActivations,
			I inFlightInput) {
		return new DirectedComponentChainActivationImpl<>(componentActivations, inFlightInput);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public GenericDirectedComponentChainImpl(List<? extends ChainableDirectedComponent<I, ? extends A, ?>> components) {
		super(components);
	}

	//@Override
	//public N dup() {
	//	return createDirectedComponentChain(components.stream().map(c -> c.dup()).collect(Collectors.toList()));
	//}
	
	//protected abstract N createDirectedComponentChain(List<ChainableDirectedComponent<I, ?, ?>> components);

	//@Override
	//protected ChainableDirectedComponentActivation<I> forwardPropagate(I input, ChainableDirectedComponent<I, ?, ?> component,
	//		int componentIndex, DirectedComponentsContext context) {
	//	return fp2(input, component, componentIndex, context);
	//}
	
	//private <C> ChainableDirectedComponentActivation<I> fp2(I input, ChainableDirectedComponent<I, ?, C> component,
	//		int componentIndex, DirectedComponentsContext context) {
	//	return component.forwardPropagate(input, getContext(context, component, componentIndex));
	//}
	
	//private <C> C getContext(DirectedComponentsContext context, ChainableDirectedComponent<I, ?, C> component,
	//		int componentIndex) {
	//	return context.getContext(component);
	//}	
}
