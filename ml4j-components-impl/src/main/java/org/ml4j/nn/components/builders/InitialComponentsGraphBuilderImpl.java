package org.ml4j.nn.components.builders;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class InitialComponentsGraphBuilderImpl extends ComponentsGraphBuilderImpl<InitialComponentsGraphBuilder> implements InitialComponentsGraphBuilder {

	public InitialComponentsGraphBuilderImpl(MatrixFactory matrixFactory, ComponentsGraphNeurons<Neurons> componentGraphNeurons, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		super(matrixFactory, componentGraphNeurons, components);
	}
	
	public InitialComponentsGraphBuilderImpl(MatrixFactory matrixFactory, Neurons initialNeurons) {
		super(matrixFactory, new ComponentsGraphNeuronsImpl<Neurons>(initialNeurons, null), new ArrayList<>());
	}

	@Override
	public InitialComponentsGraphBuilder getBuilder() {
		return this;
	}

	
}
