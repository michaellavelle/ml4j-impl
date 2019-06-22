package org.ml4j.nn.components.builders;

import java.util.List;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.axons.DirectedAxonsComponent;
import org.ml4j.nn.axons.DirectedAxonsComponentImpl;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.builders.axons.AxonsBuilder;
import org.ml4j.nn.components.builders.axons.UncompletedTrainableAxonsBuilder;
import org.ml4j.nn.components.builders.common.ComponentsContainer;
import org.ml4j.nn.components.builders.common.ComponentsParallelPathsBuilderImpl;
import org.ml4j.nn.components.builders.common.ParallelPathsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsSubGraphBuilder;
import org.ml4j.nn.components.builders.synapses.SynapsesAxonsGraphBuilder;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public abstract class ComponentsGraphBuilderImpl<C extends AxonsBuilder> implements ComponentsGraphBuilder<C>{

	protected MatrixFactory matrixFactory;
	
	private List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components;
	protected ComponentsGraphNeurons<Neurons> componentsGraphNeurons;
	
	private Matrix connectionWeights;
	
	private UncompletedTrainableAxonsBuilder<Neurons, C> fullyConnectedAxonsBuilder;
	
	private boolean initial;
	
	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> getComponents() {
		return components;
	}
	
	@Override
	public ComponentsContainer<Neurons> getAxonsBuilder() {
		return this;
	}

	public ComponentsGraphBuilderImpl(MatrixFactory matrixFactory, ComponentsGraphNeurons<Neurons> componentsGraphNeurons, List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> components) {
		this.components = components;
		this.matrixFactory = matrixFactory;
		this.componentsGraphNeurons = componentsGraphNeurons;
		if (componentsGraphNeurons == null) {
			throw new IllegalArgumentException("Components graph neurons must not be null!");
		}
		this.initial = true;
	}

	@Override
	public ComponentsGraphNeurons<Neurons> getComponentsGraphNeurons() {
		return componentsGraphNeurons;
	}

	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}

	@Override
	public AxonsBuilder withConnectionWeights(Matrix connectionWeights) {
		this.connectionWeights = connectionWeights;
		return this;
	}
	
	public abstract C getBuilder();
	
	@Override
	public UncompletedTrainableAxonsBuilder<Neurons, C> withFullyConnectedAxons() {
		
		addAxonsIfApplicable();
		this.connectionWeights = null;
		fullyConnectedAxonsBuilder = new UncompletedFullyConnectedAxonsBuilderImpl<>(getBuilder(), componentsGraphNeurons.getCurrentNeurons());
		return fullyConnectedAxonsBuilder;
	}

	public void addAxonsIfApplicable() {
		if ((initial || fullyConnectedAxonsBuilder != null) && componentsGraphNeurons.getRightNeurons() != null) {
			Neurons leftNeurons = componentsGraphNeurons.getCurrentNeurons();
			if (componentsGraphNeurons.hasBiasUnit() && !leftNeurons.hasBiasUnit()) {
				leftNeurons = new Neurons(componentsGraphNeurons.getCurrentNeurons().getNeuronCountExcludingBias(), true);
			}

			if (connectionWeights != null) {
				DirectedAxonsComponent<Neurons, Neurons> axonsComponent
				  = new DirectedAxonsComponentImpl<>(new FullyConnectedAxonsImpl(leftNeurons, componentsGraphNeurons.getRightNeurons(), matrixFactory, connectionWeights));
				this.components.add(axonsComponent);
			} else {
				DirectedAxonsComponent<Neurons, Neurons> axonsComponent
				  = new DirectedAxonsComponentImpl<>(new FullyConnectedAxonsImpl(leftNeurons, componentsGraphNeurons.getRightNeurons(), matrixFactory));
				this.components.add(axonsComponent);
			}
			
			fullyConnectedAxonsBuilder = null;
			initial= false;
			componentsGraphNeurons.setCurrentNeurons(componentsGraphNeurons.getRightNeurons());
			componentsGraphNeurons.setRightNeurons(null);
		}
		
	}

	@Override
	public SynapsesAxonsGraphBuilder<C> withSynapses() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public C withActivationFunction(DifferentiableActivationFunction activationFunction) {
		addAxonsIfApplicable();
		components.add(new DifferentiableActivationFunctionDirectedComponentImpl(activationFunction));
		return getBuilder();
	}

	@Override
	public ParallelPathsBuilder<ComponentsSubGraphBuilder<C>> withParallelPaths() {
		addAxonsIfApplicable();
		return new ComponentsParallelPathsBuilderImpl<>(matrixFactory, getBuilder());
	}
	
	public DirectedComponentChain<NeuronsActivation, ?, ?, ?> getComponentChain() {
		addAxonsIfApplicable();
		return new DefaultDirectedComponentChainImpl<>(components);
	}

	@Override
	public AxonsBuilder withBiasUnit() {
		componentsGraphNeurons.setHasBiasUnit(true);
		return this;
	}

	@Override
	public AxonsBuilder withRightNeurons(Neurons neurons) {
		componentsGraphNeurons.setRightNeurons(neurons);
		return this;
	}
}
