package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.builders.ComponentsSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ComponentsParallelPathsBuilderImpl<C extends ComponentsContainer<Neurons>> implements ParallelPathsBuilder<ComponentsSubGraphBuilder<C>> {

	private MatrixFactory matrixFactory;
	private C previous;
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> subGraphComponents;
	private ComponentsSubGraphBuilder<C> currentPath;
	
	public ComponentsParallelPathsBuilderImpl(MatrixFactory matrixFactory, C previous) {
		this.matrixFactory = matrixFactory;
		this.previous = previous;
		this.subGraphComponents = new ArrayList<>();
	}
	
	@Override
	public ComponentsSubGraphBuilder<C> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath =  new ComponentsSubGraphBuilderImpl<>(previous, matrixFactory, previous.getComponentsGraphNeurons(), new ArrayList<>(), subGraphComponents);
		return currentPath;
	}
}
