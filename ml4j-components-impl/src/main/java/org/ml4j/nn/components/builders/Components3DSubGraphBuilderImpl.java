package org.ml4j.nn.components.builders;

import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.builders.common.Components3DSubGraphPathEnder;
import org.ml4j.nn.components.builders.common.PathEnder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public class Components3DSubGraphBuilderImpl<P extends Components3DGraphBuilder<P, Q>, Q extends ComponentsGraphBuilder<Q>>
		extends Components3DGraphBuilderImpl<Components3DSubGraphBuilder<P, Q>, ComponentsSubGraphBuilder<Q>>
		implements Components3DSubGraphBuilder<P, Q> {

	private P previous;
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> subGraphComponents;
	private ComponentsSubGraphBuilder<Q> builder;

	public Components3DSubGraphBuilderImpl(P previous, MatrixFactory matrixFactory,
			ComponentsGraphNeurons<Neurons3D> neurons,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components,
			List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> subGraphComponents) {
		super(matrixFactory, neurons, components);
		this.previous = previous;
		this.subGraphComponents = subGraphComponents;
	}

	@Override
	public PathEnder<P, Components3DSubGraphBuilder<P, Q>> endPath() {
		//System.out.println("Ending path");
		return new Components3DSubGraphPathEnder<P, Q>(previous, this, () -> new Components3DSubGraphBuilderImpl<>(previous,
				matrixFactory, getComponentsGraphNeurons(), getComponents(), subGraphComponents));
	}

	@Override
	public Components3DSubGraphBuilder<P, Q> get3DBuilder() {
		return this;
	}

	@Override
	public ComponentsSubGraphBuilder<Q> getBuilder() {
		if (builder != null) {
			return builder;
		} else {

			Neurons currentNeurons = null;
			Neurons rightNeurons = null;
			if (getComponentsGraphNeurons().getCurrentNeurons() != null) {
				currentNeurons = new Neurons(
						getComponentsGraphNeurons().getCurrentNeurons().getNeuronCountExcludingBias(), false);
			}
			if (getComponentsGraphNeurons().getRightNeurons() != null) {
				rightNeurons = new Neurons(getComponentsGraphNeurons().getRightNeurons().getNeuronCountExcludingBias(),
						false);
			}
			return new ComponentsSubGraphBuilderImpl<>(previous.getBuilder(), matrixFactory,
					new ComponentsGraphNeuronsImpl<Neurons>(currentNeurons, rightNeurons), getComponents(),
					subGraphComponents);
		}
	}

}
