package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.axons.AxonsGradient;

public class DirectedComponentBatchActivationImpl<I, A extends DirectedComponentActivation<I, I>> implements DirectedComponentBatchActivation<I, A> {

	private List<A> activations;
	private List<I> output;
	
	public DirectedComponentBatchActivationImpl(List<A> activations) {
		this.activations = activations;
		this.output = activations.stream().map(DirectedComponentActivation::getOutput).collect(Collectors.toList());
	}
	
	@Override
	public List<A> getActivations() {
		return activations;
	}


	@Override
	public DirectedComponentGradient<List<I>> backPropagate(DirectedComponentGradient<List<I>> outerGradient) {
		int index = 0;
		List<AxonsGradient> allAxonsGradients = new ArrayList<>();
		List<I> combinedOutput = new ArrayList<>();
		for (A activation : activations) {
			DirectedComponentGradient<I> grad = new DirectedComponentGradientImpl<>(outerGradient.getOutput().get(index));
			DirectedComponentGradient<I> backPropGrad = activation.backPropagate(grad);
			combinedOutput.add(backPropGrad.getOutput());
			List<AxonsGradient> backPropAxonsGradients = backPropGrad.getTotalTrainableAxonsGradients();
			allAxonsGradients.addAll(backPropAxonsGradients);
			index++;
		}
		return new DirectedComponentGradientImpl<>(allAxonsGradients, combinedOutput);
	}


	@Override
	public List<I> getOutput() {
		return output;
	}
}
