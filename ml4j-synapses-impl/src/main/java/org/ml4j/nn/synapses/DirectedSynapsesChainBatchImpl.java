package org.ml4j.nn.synapses;

import java.util.List;

import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;

public class DirectedSynapsesChainBatchImpl<S extends DirectedSynapses<?, ?>> extends DefaultDirectedComponentChainBatchImpl<DirectedSynapsesChain<S>, DirectedSynapsesChainActivation, DirectedSynapsesActivation>
 implements DirectedSynapsesChainBatch<S> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DirectedSynapsesChainBatchImpl(List<DirectedSynapsesChain<S>> components) {
		super(components);
	}

}
