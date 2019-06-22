package org.ml4j.nn.components.builders;

import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedTrainableAxonsBuilder;
import org.ml4j.nn.neurons.Neurons3D;

public class UncompletedConvolutionalAxonsBuilderImpl<C extends Axons3DBuilder> extends UncompletedTrainableAxonsBuilderImpl<Neurons3D, C> {

	public UncompletedConvolutionalAxonsBuilderImpl(C previousBuilder, Neurons3D leftNeurons) {
		super(previousBuilder, leftNeurons);
	}

	@Override
	public C withConnectionToNeurons(Neurons3D neurons) {
		previousBuilder.withRightNeurons(neurons);
		return previousBuilder;
	}
	
	@Override
	public UncompletedTrainableAxonsBuilder<Neurons3D, C> withBiasUnit() {
		previousBuilder.withBiasUnit();
		return super.withBiasUnit();
	}
}
