package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

public abstract class DirectedComponentBatchImpl<I, L extends DirectedComponent<I, A, C>, A extends DirectedComponentActivation<I, I>, C, C2> implements DirectedComponentBatch<I, L, DirectedComponentBatchActivation<I, A>, A,  C, C2> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	private List<L> components;
	
	public DirectedComponentBatchImpl(List<L> components) {
		this.components = components;
	}
	
	@Override
	public List<L> getComponents() {
		return components;
	}

	@Override
	public DirectedComponentBatchActivation<I, A> forwardPropagate(List<I> input, C2 context) {
		
		int index = 0;
		List<A> activations = new ArrayList<>();
		for (L component : components) {
			A activation = component.forwardPropagate(input.get(index), getContext(context, component, index));
			activations.add(activation);
			index++;
		}
		
		return new DirectedComponentBatchActivationImpl<>(activations);
	}
	
	protected abstract C getContext(C2 context, L component, int index);
}
