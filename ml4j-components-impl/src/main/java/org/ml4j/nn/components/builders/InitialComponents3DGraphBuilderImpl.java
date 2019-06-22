package org.ml4j.nn.components.builders;

import java.util.ArrayList;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponents3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.InitialComponentsGraphBuilder;
import org.ml4j.nn.neurons.Neurons3D;

public class InitialComponents3DGraphBuilderImpl extends Components3DGraphBuilderImpl<InitialComponents3DGraphBuilder, InitialComponentsGraphBuilder> implements InitialComponents3DGraphBuilder{
	
	private InitialComponentsGraphBuilder nestedBuilder;
	
	public InitialComponents3DGraphBuilderImpl(MatrixFactory matrixFactory, Neurons3D currentNeurons) {
		super(matrixFactory, new ComponentsGraphNeuronsImpl<>(currentNeurons), new ArrayList<>());
	}

	@Override
	public InitialComponents3DGraphBuilder get3DBuilder() {
		return this;
	}

	@Override
	public InitialComponentsGraphBuilder getBuilder() {
		if (nestedBuilder != null) {
			return nestedBuilder;
		} else {
			nestedBuilder = new InitialComponentsGraphBuilderImpl(matrixFactory, new ComponentsGraphNeuronsImpl<>(getComponentsGraphNeurons().getCurrentNeurons(), getComponentsGraphNeurons().getRightNeurons()), getComponents());
			return nestedBuilder;
		}
	}
}
