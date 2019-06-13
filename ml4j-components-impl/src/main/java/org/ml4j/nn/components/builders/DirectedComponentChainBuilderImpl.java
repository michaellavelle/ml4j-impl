package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedComponentChainBuilderImpl implements DirectedComponentChainBuilder {
	
	private Neurons currentNeurons;
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> componentList;
	private DirectedComponentsContext context;
	
	public DirectedComponentChainBuilderImpl(Neurons currentNeurons, DirectedComponentsContext context) {
		this.currentNeurons = currentNeurons;
		this.componentList = new ArrayList<>();
		this.context = context;
	}

	
	public void setCurrentNeurons(Neurons currentNeurons) {
		this.currentNeurons = currentNeurons;
	}

	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> getComponentList() {
		return componentList;
	}


	@Override
	public DirectedComponentChain<NeuronsActivation, ?, ?, ?> build() {
		return new DefaultDirectedComponentChainImpl<>(componentList);
	}


	@Override
	public DirectedAxonsComponentBuilder<DirectedComponentChainBuilder> withFullyConnectedAxons() {
		if (!currentNeurons.hasBiasUnit()) {
			currentNeurons = new Neurons(currentNeurons.getNeuronCountExcludingBias(), true);
		}
		return new DirectedAxonsComponentBuilderImpl<>(context, currentNeurons, this);
	}

	@Override
	public DirectedComponentChainBuilder withActivationFunction(DifferentiableActivationFunction activationFunction) {
		componentList.add(new DifferentiableActivationFunctionDirectedComponentImpl(activationFunction));
		return this;
	}
}
