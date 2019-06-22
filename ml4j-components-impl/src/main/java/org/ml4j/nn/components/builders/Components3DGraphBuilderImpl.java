package org.ml4j.nn.components.builders;

import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.axons.DirectedAxonsComponent;
import org.ml4j.nn.axons.DirectedAxonsComponentImpl;
import org.ml4j.nn.axons.MaxPoolingAxonsImpl;
import org.ml4j.nn.axons.UnpaddedConvolutionalAxonsImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.builders.axons.Axons3DBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedAxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedTrainableAxonsBuilder;
import org.ml4j.nn.components.builders.common.Components3DParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.Components3DSubGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.SynapsesAxons3DGraphBuilder;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class Components3DGraphBuilderImpl<C extends Components3DGraphBuilder<C, D>, D extends ComponentsGraphBuilder<D>> implements Components3DGraphBuilder<C, D> {

	protected MatrixFactory matrixFactory;
	
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components;

	private ComponentsGraphNeurons<Neurons3D> componentsGraphNeurons;
	private UncompletedTrainableAxonsBuilder<Neurons3D, C> convolutionalAxonsBuilder;
	private UncompletedAxonsBuilder<Neurons3D, C> maxPoolingAxonsBuilder;
	
	public abstract C get3DBuilder();
	public abstract D getBuilder();
	
	
	@Override
	public ComponentsContainer<Neurons> getAxonsBuilder() {
		return getBuilder();
	}
	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> getComponents() {
		return components;
	}
	
	@Override
	public ComponentsGraphNeurons<Neurons3D> getComponentsGraphNeurons() {
		return componentsGraphNeurons;
	}
	
	public Components3DGraphBuilderImpl(MatrixFactory matrixFactory, ComponentsGraphNeurons<Neurons3D> componentsGraphNeurons, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		this.components = components;
		this.matrixFactory = matrixFactory;
		this.componentsGraphNeurons = componentsGraphNeurons;
	}
	
	public void addAxonsIfApplicable() {
		if (convolutionalAxonsBuilder != null && componentsGraphNeurons.getRightNeurons() != null) {

			Neurons3D leftNeurons = componentsGraphNeurons.getCurrentNeurons();
			if (componentsGraphNeurons.hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons3D(componentsGraphNeurons.getCurrentNeurons().getWidth(), componentsGraphNeurons.getCurrentNeurons().getHeight(), componentsGraphNeurons.getCurrentNeurons().getDepth(), true);
			}
			if (convolutionalAxonsBuilder.getConnectionWeights() != null) {
				DirectedAxonsComponent<Neurons, Neurons> axonsComponent
				  = new DirectedAxonsComponentImpl<>(new UnpaddedConvolutionalAxonsImpl(leftNeurons, componentsGraphNeurons.getRightNeurons(), 1, matrixFactory, convolutionalAxonsBuilder.getConnectionWeights()));
				this.components.add(axonsComponent);
			} else {
				DirectedAxonsComponent<Neurons, Neurons> axonsComponent
				  = new DirectedAxonsComponentImpl<>(new UnpaddedConvolutionalAxonsImpl(leftNeurons, componentsGraphNeurons.getRightNeurons(), 1, matrixFactory));
				this.components.add(axonsComponent);
			}
		
			convolutionalAxonsBuilder = null;
			componentsGraphNeurons.setCurrentNeurons(componentsGraphNeurons.getRightNeurons());
			componentsGraphNeurons.setRightNeurons(null);
		} 
		if (maxPoolingAxonsBuilder != null && componentsGraphNeurons.getRightNeurons() != null) {

			DirectedAxonsComponent<Neurons3D, Neurons3D> axonsComponent
			  = new DirectedAxonsComponentImpl<>(new MaxPoolingAxonsImpl(componentsGraphNeurons.getCurrentNeurons(), componentsGraphNeurons.getRightNeurons(), matrixFactory, true));
			this.components.add(axonsComponent);

			maxPoolingAxonsBuilder = null;
			componentsGraphNeurons.setCurrentNeurons(componentsGraphNeurons.getRightNeurons());
			componentsGraphNeurons.setRightNeurons(null);
		}
	}
	
	@Override
	public Axons3DBuilder withRightNeurons(Neurons3D neurons) {
		componentsGraphNeurons.setRightNeurons(neurons);
		return this;
	}


	@Override
	public Axons3DBuilder withBiasUnit() {
		componentsGraphNeurons.setHasBiasUnit(true);
		return this;
	}
	@Override
	public UncompletedTrainableAxonsBuilder<Neurons, D> withFullyConnectedAxons() {
		addAxonsIfApplicable();
		return new UncompletedFullyConnectedAxonsBuilderImpl<>(getBuilder(), componentsGraphNeurons.getCurrentNeurons());
	}

	@Override
	public UncompletedTrainableAxonsBuilder<Neurons3D, C> withConvolutionalAxons() {
		addAxonsIfApplicable();
		this.convolutionalAxonsBuilder = new UncompletedConvolutionalAxonsBuilderImpl<>(get3DBuilder(), componentsGraphNeurons.getCurrentNeurons());
		return convolutionalAxonsBuilder;
	}

	@Override
	public UncompletedAxonsBuilder<Neurons3D, C> withMaxPoolingAxons() {
		addAxonsIfApplicable();
		this.maxPoolingAxonsBuilder = new UncompletedMaxPoolingAxonsBuilderImpl<>(get3DBuilder(), componentsGraphNeurons.getCurrentNeurons());
		return maxPoolingAxonsBuilder;
	}

	@Override
	public SynapsesAxons3DGraphBuilder<C> withSynapses() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addAxonsIfApplicable();
		components.add(new DifferentiableActivationFunctionDirectedComponentImpl(activationFunction));
		return get3DBuilder();
	}

	@Override
	public ParallelPathsBuilder<Components3DSubGraphBuilder<C, D>> withParallelPaths() {
		//System.out.println("Starting parallelPaths");
		addAxonsIfApplicable();
		return new Components3DParallelPathsBuilderImpl<>(matrixFactory, get3DBuilder());

	}

	public DirectedComponentChain<NeuronsActivation, ?, ?, ?> getComponentChain() {
		addAxonsIfApplicable();
		return new DefaultDirectedComponentChainImpl<>(components);
	}
}
