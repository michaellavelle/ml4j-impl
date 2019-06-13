package org.ml4j.nn.synapses;

import java.util.function.IntSupplier;

import org.ml4j.nn.components.DirectedComponentBatch;
import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.ManyToOneDirectedComponent;
import org.ml4j.nn.components.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.OneToManyDirectedComponent;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.defaults.DefaultManyToOneDirectedComponent;
import org.ml4j.nn.components.defaults.DefaultOneToManyDirectedComponent;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedSynapsesBipoleGraphImpl<S extends DirectedSynapses<?, ?>> extends DirectedComponentsBipoleGraphImpl<NeuronsActivation, DirectedSynapsesChain<S>, DirectedComponentBatch<NeuronsActivation, DirectedSynapsesChain<S> , DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation>, DirectedSynapsesChainActivation, DirectedComponentsContext, DirectedComponentsContext>, DirectedSynapsesChainActivation, DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation>, DirectedSynapsesBipoleGraphActivation, DirectedComponentsContext, DirectedComponentsContext>
	implements DirectedSynapsesBipoleGraph<S> {


	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private boolean includesBias;

	public DirectedSynapsesBipoleGraphImpl(
			DirectedComponentBatch<NeuronsActivation, DirectedSynapsesChain<S>, DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation>, DirectedSynapsesChainActivation, DirectedComponentsContext, 
			DirectedComponentsContext> edges, boolean includesBias) {
		super(edges);
	}

	@Override
	protected DirectedSynapsesBipoleGraphActivation createActivation(
			OneToManyDirectedComponentActivation<NeuronsActivation> oneToManyActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation> batchActivation,
			ManyToOneDirectedComponentActivation<NeuronsActivation> manyToOneAct) {
		return new DirectedSynapsesBipoleGraphActivationImpl(oneToManyActivation, batchActivation, manyToOneAct);

	}

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
