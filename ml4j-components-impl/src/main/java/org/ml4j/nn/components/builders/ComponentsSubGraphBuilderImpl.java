package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ComponentsSubGraphPathEnder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ComponentsSubGraphBuilderImpl<P extends ComponentsContainer<Neurons>> extends ComponentsGraphBuilderImpl<ComponentsSubGraphBuilder<P>> implements ComponentsSubGraphBuilder<P> {

	private P previous;
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> subGraphComponents;
	
	public ComponentsSubGraphBuilderImpl(P previous, MatrixFactory matrixFactory, ComponentsGraphNeurons<Neurons> componentsGraphNeurons, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> subGraphComponents) {
		super(matrixFactory, componentsGraphNeurons, components);
		this.previous = previous;
		this.subGraphComponents = subGraphComponents;
	}

	@Override
	public PathEnder<P, ComponentsSubGraphBuilder<P>> endPath() {
		//System.out.println("Ending path");
		return new ComponentsSubGraphPathEnder<> (previous, subGraphComponents, this,  () -> new ComponentsSubGraphBuilderImpl<P>(previous, matrixFactory, componentsGraphNeurons, new ArrayList<>(), subGraphComponents));
	}

	@Override
	public ComponentsSubGraphBuilder<P> getBuilder() {
		return this;
	}

}
