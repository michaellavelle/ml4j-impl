package org.ml4j.nn.components.builders;

import org.ml4j.Matrix;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedTrainableAxonsBuilder;
import org.ml4j.nn.neurons.Neurons;

public class UncompletedFullyConnectedAxonsBuilderImpl<C extends AxonsBuilder> extends UncompletedTrainableAxonsBuilderImpl<Neurons, C> {

	public UncompletedFullyConnectedAxonsBuilderImpl(C previousBuilder, Neurons leftNeurons) {
		super(previousBuilder, leftNeurons);
	}

	@Override
	public C withConnectionToNeurons(Neurons neurons) {
		previousBuilder.withRightNeurons(neurons);
		return previousBuilder;
	}

	@Override
	public UncompletedTrainableAxonsBuilder<Neurons, C> withConnectionWeights(Matrix connectionWeights) {
		previousBuilder.withConnectionWeights(connectionWeights);
		return super.withConnectionWeights(connectionWeights);
	}

	@Override
	public UncompletedTrainableAxonsBuilder<Neurons, C> withBiasUnit() {
		previousBuilder.withBiasUnit();
		return super.withBiasUnit();
	}
	
	

}
