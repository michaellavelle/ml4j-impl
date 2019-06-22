package org.ml4j.nn.components.builders;

import org.ml4j.Matrix;
import org.ml4j.nn.components.builders.axons.UncompletedTrainableAxonsBuilder;
import org.ml4j.nn.neurons.Neurons;

public abstract class UncompletedTrainableAxonsBuilderImpl<N extends Neurons, C> extends UncompletedAxonsBuilderImpl<N, C>
		implements UncompletedTrainableAxonsBuilder<N, C> {
	
	private boolean withBiasUnit;
	private Matrix connectionWeights;
	
	public UncompletedTrainableAxonsBuilderImpl(C previousBuilder, N leftNeurons) {
		super(previousBuilder, leftNeurons);
	}

	@Override
	public UncompletedTrainableAxonsBuilder<N, C> withConnectionWeights(Matrix connectionWeights) {
		this.connectionWeights = connectionWeights;
		return this;
	}

	@Override
	public UncompletedTrainableAxonsBuilder<N, C> withBiasUnit() {
		this.withBiasUnit = true;
		return this;
	}
	
	@Override
	public boolean isWithBiasUnit() {
		return withBiasUnit;
	}
	
	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}
}
