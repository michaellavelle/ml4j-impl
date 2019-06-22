package org.ml4j.nn.components.builders.common;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.builders.Components3DSubGraphBuilderImpl;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.neurons.NeuronsActivation;

public class Components3DParallelPathsBuilderImpl<C extends Components3DGraphBuilder<C, D>, D extends ComponentsGraphBuilder<D>> implements ParallelPathsBuilder<Components3DSubGraphBuilder<C, D>> {

	private MatrixFactory matrixFactory;
	private C previous;
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> subGraphComponents;
	private Components3DSubGraphBuilder<C, D> currentPath;
	
	public Components3DParallelPathsBuilderImpl(MatrixFactory matrixFactory, C previous) {
		this.matrixFactory = matrixFactory;
		this.previous = previous;
		this.subGraphComponents = new ArrayList<>();
	}
	
	@Override
	public Components3DSubGraphBuilder<C, D> withPath() {
		if (currentPath != null) {
			throw new UnsupportedOperationException("Multiple paths not yet supported");
		}
		currentPath = new Components3DSubGraphBuilderImpl<>(previous,  matrixFactory, previous.getComponentsGraphNeurons(), new ArrayList<>(), subGraphComponents);
		return currentPath;
	}
	
	
}
