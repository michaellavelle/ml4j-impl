package org.ml4j.nn.components.builders.common;

import java.util.List;
import java.util.function.Supplier;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ComponentsSubGraphPathEnder<P extends ComponentsContainer<Neurons>> extends PathEnderImpl<P, ComponentsSubGraphBuilder<P>> implements PathEnder<P, ComponentsSubGraphBuilder<P>> {

	
	private ComponentsSubGraphBuilder<P> subGraphBuilder;
	private ComponentsSubGraphBuilder<P> unprocessedBuilder;
	
	public ComponentsSubGraphPathEnder(P previous, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> subGraphComponents, ComponentsSubGraphBuilder<P> subGraphBuilder, Supplier<ComponentsSubGraphBuilder<P>> newPathCreator) {
		super(previous, newPathCreator);
		this.subGraphBuilder = subGraphBuilder;
		this.unprocessedBuilder = subGraphBuilder;
	}
	
	private void processPath() {

		if (unprocessedBuilder != null) {
		subGraphBuilder.addAxonsIfApplicable();
		
		// TODO ML Here we would add the components to parallel paths instead of adding serially
		previous.getComponents().addAll(subGraphBuilder.getComponents());
		previous.getComponentsGraphNeurons().setCurrentNeurons(subGraphBuilder.getComponentsGraphNeurons().getCurrentNeurons());
		previous.getComponentsGraphNeurons().setRightNeurons(subGraphBuilder.getComponentsGraphNeurons().getRightNeurons());
		unprocessedBuilder = null;
		
		}
	}

	@Override
	protected void onParallelPathsEnd() {
		processPath();
	}
	
	@Override
	protected void onPathEnd() {
		processPath();
	}

	
}
