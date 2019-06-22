package org.ml4j.nn.components.builders.common;

import java.util.function.Supplier;

import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;

public class Components3DSubGraphPathEnder<P extends ComponentsContainer<Neurons3D>, Q extends ComponentsContainer<Neurons>> extends PathEnderImpl<P, Components3DSubGraphBuilder<P, Q>> implements PathEnder<P, Components3DSubGraphBuilder<P, Q>> {
	
	private Components3DSubGraphBuilder<P, Q> subGraphBuilder;
	private Components3DSubGraphBuilder<P, Q> unprocessedBuilder;

	public Components3DSubGraphPathEnder(P previous, Components3DSubGraphBuilder<P, Q> subGraphBuilder, Supplier<Components3DSubGraphBuilder<P, Q>> newPathCreator) {
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
			previous.getComponentsGraphNeurons().setHasBiasUnit(subGraphBuilder.getComponentsGraphNeurons().hasBiasUnit());
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
