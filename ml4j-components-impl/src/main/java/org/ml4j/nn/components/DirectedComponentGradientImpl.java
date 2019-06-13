package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.axons.AxonsGradient;

public class DirectedComponentGradientImpl<O> implements DirectedComponentGradient<O> {

	private O output;
	private List<AxonsGradient> totalTrainableAxonsGradients;
	private List<AxonsGradient> averageTrainableAxonsGradients;

	public DirectedComponentGradientImpl(List<AxonsGradient> totalTrainableAxonsGradients, AxonsGradient axonsGradient, O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.totalTrainableAxonsGradients.addAll(totalTrainableAxonsGradients);
		this.totalTrainableAxonsGradients.add(axonsGradient);
		this.output = output;
	}
	
	public DirectedComponentGradientImpl(List<AxonsGradient> totalTrainableAxonsGradients, O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.totalTrainableAxonsGradients.addAll(totalTrainableAxonsGradients);
		this.output = output;
	}
	
	public DirectedComponentGradientImpl(DirectedComponentGradient<?> previousGradient, O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.totalTrainableAxonsGradients.addAll(previousGradient.getTotalTrainableAxonsGradients());
		this.output = output;
	}

	
	public DirectedComponentGradientImpl(O output) {
		this.totalTrainableAxonsGradients = new ArrayList<>();
		this.output = output;
	}

	@Override
	public O getOutput() {
		return output;
	}

	@Override
	public void addTotalTrainableAxonsGradient(AxonsGradient axonsGradient) {
		totalTrainableAxonsGradients.add(axonsGradient);
	}

	@Override
	public List<AxonsGradient> getTotalTrainableAxonsGradients() {
		return totalTrainableAxonsGradients;
	}

	@Override
	public List<AxonsGradient> getAverageTrainableAxonsGradients() {
		return averageTrainableAxonsGradients;
	}
}
