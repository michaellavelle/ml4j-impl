package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntSupplier;

public abstract class OneToManyDirectedComponent<I, C> implements DirectedComponent<I, OneToManyDirectedComponentActivation<I>, C> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private IntSupplier outputCount;
	
	public OneToManyDirectedComponent(IntSupplier outputCount) {
		this.outputCount = outputCount;
	}
	
	@Override
	public OneToManyDirectedComponentActivation<I> forwardPropagate(I input,
			C synapsesContext) {
		List<I> acts = new ArrayList<>();
		for (int i = 0; i < outputCount.getAsInt(); i++) {
			acts.add(input);
		}
		
		return createActivation(acts, synapsesContext);
	}

	protected abstract OneToManyDirectedComponentActivation<I> createActivation(List<I> acts, C synapsesContext);

}
