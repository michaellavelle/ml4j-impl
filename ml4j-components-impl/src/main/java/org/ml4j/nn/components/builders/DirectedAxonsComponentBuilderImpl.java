package org.ml4j.nn.components.builders;

import java.util.function.Consumer;

import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.DirectedAxonsComponent;
import org.ml4j.nn.axons.DirectedAxonsComponentImpl;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons;

public class DirectedAxonsComponentBuilderImpl<C extends DirectedComponentChainBuilder> implements DirectedAxonsComponentBuilder <C> {
	
	private C context;
	private Neurons initialNeurons;
	private DirectedComponentsContext directedComponentsContext;
	
	public DirectedAxonsComponentBuilderImpl(DirectedComponentsContext directedComponentsContext, Neurons initialNeurons, C context) {
		this.context = context;
		this.initialNeurons = initialNeurons;
		this.directedComponentsContext = directedComponentsContext;
	}

	@Override
	public C withConnectionToNeurons(Neurons neurons, Consumer<AxonsContext>... contextConfigurers) {
		FullyConnectedAxons axons = new FullyConnectedAxonsImpl(initialNeurons, neurons, directedComponentsContext.getMatrixFactory());
		DirectedAxonsComponent<?, ?> component = new DirectedAxonsComponentImpl<>(axons);
		AxonsContext axonsContext = component.getContext(directedComponentsContext, 0);
		for (Consumer<AxonsContext> configurer : contextConfigurers) {
			configurer.accept(axonsContext);
		}
		context.getComponentList().add(component);
		context.setCurrentNeurons(neurons);
		return context;
	}

	@Override
	public Neurons getInitialNeurons() {
		return initialNeurons;
	}

}
