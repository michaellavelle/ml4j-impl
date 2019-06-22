package org.ml4j.nn.components.builders;

import org.ml4j.nn.components.builders.axons.UncompletedAxonsBuilder;
import org.ml4j.nn.neurons.Neurons;

public abstract class UncompletedAxonsBuilderImpl<N extends Neurons, C> implements UncompletedAxonsBuilder<N, C> {

	protected C previousBuilder;
	protected N leftNeurons;
	
	public UncompletedAxonsBuilderImpl(C previousBuilder, N leftNeurons) {
		this.previousBuilder = previousBuilder;
		this.leftNeurons = leftNeurons;
	}
	
	public N getLeftNeurons() {
		return leftNeurons;
	}
}
