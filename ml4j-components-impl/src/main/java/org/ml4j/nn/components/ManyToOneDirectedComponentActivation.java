package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

public class ManyToOneDirectedComponentActivation<I> implements DirectedComponentActivation<List<I>, I> {

	private I output;
	private int inputCount;
	
	public ManyToOneDirectedComponentActivation(I output, int inputCount) { 
		this.output = output;
		this.inputCount = inputCount;
	}
	
	@Override
	public DirectedComponentGradient<List<I>> backPropagate(DirectedComponentGradient<I> outerGradient) {
		List<I> outputs = new ArrayList<>();
		for (int i = 0; i < inputCount; i++) {
			outputs.add(outerGradient.getOutput());
		}
		return new DirectedComponentGradientImpl<>(outerGradient.getTotalTrainableAxonsGradients(), outputs);
	}

	@Override
	public I getOutput() {
		return output;
	}

}
