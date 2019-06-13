package org.ml4j.nn.components.defaults;

import java.util.function.IntSupplier;

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentChainBatch;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.GenericDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.ManyToOneDirectedComponent;
import org.ml4j.nn.components.OneToManyDirectedComponent;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentChainBipoleGraphImpl<CH extends DirectedComponentChain<NeuronsActivation, ?, ?, CHA>, CHA extends DirectedComponentChainActivation<NeuronsActivation, ?>> extends GenericDirectedComponentChainBipoleGraph<NeuronsActivation, CH, CHA> {

	
	public DefaultDirectedComponentChainBipoleGraphImpl(
			DirectedComponentChainBatch<NeuronsActivation, CH, CHA, DirectedComponentBatchActivation<NeuronsActivation, CHA>> edges) {
		super(edges);
	}

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;



	@Override
	protected OneToManyDirectedComponent<NeuronsActivation, DirectedComponentsContext> createOneToManyDirectedComponent(
			IntSupplier size) {
		return new DefaultOneToManyDirectedComponent(size);
	}

	@Override
	protected ManyToOneDirectedComponent<NeuronsActivation, DirectedComponentsContext> createManyToOneDirectedComponent() {
		return new DefaultManyToOneDirectedComponent();
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext,
			int componentIndex) {
		return directedComponentsContext;
	}
	
}
