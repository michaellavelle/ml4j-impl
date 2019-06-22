package org.ml4j.nn.components.builders;

import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.neurons.Neurons3D;

public class UncompletedMaxPoolingAxonsBuilderImpl<C extends Axons3DBuilder> extends UncompletedAxonsBuilderImpl<Neurons3D, C> {

	public UncompletedMaxPoolingAxonsBuilderImpl(C previousBuilder, Neurons3D leftNeurons) {
		super(previousBuilder, leftNeurons);
	}

	@Override
	public C withConnectionToNeurons(Neurons3D neurons) {
		previousBuilder.withRightNeurons(neurons);
		return previousBuilder;
	}


}
