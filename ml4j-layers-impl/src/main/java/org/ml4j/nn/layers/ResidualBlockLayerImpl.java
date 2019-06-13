package org.ml4j.nn.layers;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionDirectedComponentImpl;
import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.DirectedAxonsComponent;
import org.ml4j.nn.axons.DirectedAxonsComponentImpl;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.axons.FullyConnectedAxonsImpl;
import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentChain;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChain;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBatchImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainBipoleGraphImpl;
import org.ml4j.nn.components.defaults.DefaultDirectedComponentChainImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class ResidualBlockLayerImpl extends AbstractFeedForwardLayer<Axons<?, ?, ?>, ResidualBlockLayerImpl> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private FeedForwardLayer<?, ?> layer1;
	private FeedForwardLayer<?, ?> layer2;
	
	public ResidualBlockLayerImpl(FeedForwardLayer<?, ?> layer1, FeedForwardLayer<?, ?> layer2, 
			MatrixFactory matrixFactory) {
		super(createComponentChain(layer1, layer2, matrixFactory), matrixFactory);
		this.layer1 = layer1;
		this.layer2 = layer2;
	}
	
	private static DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>> createPrecedingChain(FeedForwardLayer<?, ?> layer1, FeedForwardLayer<?, ?> layer2) {
		
		 // Start with all components
		 List<ChainableDirectedComponent<NeuronsActivation, ?, ?>> allComponents = new ArrayList<>();
		 allComponents.addAll(layer1.getComponents());
		 allComponents.addAll(layer2.getComponents());
		 
		 // Set preceedingComponents list to have all but the last synapses
		 List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> preceedingComponents = new ArrayList<>();
		 for (ChainableDirectedComponent<NeuronsActivation, ?, ?> comp : allComponents.subList(0, allComponents.size() -1)) {
			 preceedingComponents.add(comp);
		 }
		 
		 // Create an axons only component from the last synapses
		 DirectedAxonsComponent<Neurons, Neurons> axonsComponent = new DirectedAxonsComponentImpl<Neurons, Neurons>(layer2.getPrimaryAxons());
		 preceedingComponents.add(axonsComponent);
		 
		 // Return the component chain consisting of all components except the last activation function
		 return new DefaultDirectedComponentChainImpl<>(preceedingComponents);
		
	}

	private static DirectedComponentChain<NeuronsActivation, ? extends ChainableDirectedComponent<NeuronsActivation, ?, ?>, ?, ?> createComponentChain(
			FeedForwardLayer<?, ?> layer1, FeedForwardLayer<?, ?> layer2, MatrixFactory matrixFactory) {
	
		 // Final activation function component
		 DifferentiableActivationFunctionComponent finalActivationFunctionComponent = new DifferentiableActivationFunctionDirectedComponentImpl(layer2.getPrimaryActivationFunction());
		 
		 // Chain of components before the final activation function component
		 DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>> precedingChain 
				 = createPrecedingChain(layer1, layer2);
		 
		 List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> matchingAxonsList = new ArrayList<>();
		 
		 // If the layer sizes don't match up, create axons to match the two sizes and add to matchingAxonsList
		 if (layer1.getInputNeuronCount() != (layer2.getOutputNeuronCount() + 1)) {
			 
			 FullyConnectedAxons matchingAxons = new FullyConnectedAxonsImpl(layer1.getPrimaryAxons().getLeftNeurons(), layer2.getPrimaryAxons().getRightNeurons(), matrixFactory);
			 DirectedAxonsComponent<Neurons, Neurons> matchingComponent = new DirectedAxonsComponentImpl<>(matchingAxons);
			 matchingAxonsList.add(matchingComponent);
		 } 
		 
		 // Skip connection chain, either empty or containing the matching axons
		DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>> skipConnectionChain 
				 = new DefaultDirectedComponentChainImpl<>(matchingAxonsList);
		 
		// Parallel Chains of preceding chain and skip connection
		List<DefaultDirectedComponentChain<ChainableDirectedComponentActivation<NeuronsActivation>>> parallelChains = new ArrayList<>();
		parallelChains.add(precedingChain);
		parallelChains.add(skipConnectionChain);
		
		// Parallel Chain Batch of preceding chain and skip connection
		DefaultDirectedComponentChainBatch<?, ?> parallelBatch =  new DefaultDirectedComponentChainBatchImpl<>(parallelChains);
	
		// Parallel Chain Graph of preceding chain and skip connection
		DefaultDirectedComponentChainBipoleGraphImpl<?, ?> parallelGraph = new DefaultDirectedComponentChainBipoleGraphImpl<>(parallelBatch);						
		
		// Residual block component list is composed of the parallel chain graph followed by the final activation function
		//List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> residualBlockListOfComponents 
		//	= Arrays.asList(parallelGraph, finalActivationFunctionComponent);
		 
		// Create a DirectedComponentChain from the list of components, that has an activation function as the final component
	//	return new TrailingActivationFunctionDirectedComponentChainImpl<>(residualBlockListOfComponents);
		return null;
	}

	@Override
	public int getInputNeuronCount() {
		return layer1.getInputNeuronCount();
	}

	@Override
	public int getOutputNeuronCount() {
		return layer2.getOutputNeuronCount();
	}

	@Override
	public DifferentiableActivationFunction getPrimaryActivationFunction() {
		throw new UnsupportedOperationException("Not supported");
	}

	@Override
	public NeuronsActivation getOptimalInputForOutputNeuron(int outpuNeuronIndex,
			DirectedLayerContext directedLayerContext) {
		return layer2.getOptimalInputForOutputNeuron(outpuNeuronIndex, directedLayerContext);
	}

	@Override
	public ResidualBlockLayerImpl dup() {
		return new ResidualBlockLayerImpl(layer1.dup(), layer2.dup(), matrixFactory);
	}

	@Override
	public Axons<?, ?, ?> getPrimaryAxons() {
		throw new UnsupportedOperationException("Not supported");
	}

	@Override
	public List<ChainableDirectedComponent<NeuronsActivation, ?, ?>> getComponents() {
		List<ChainableDirectedComponent<NeuronsActivation, ?, ?>> components =  new ArrayList<>();
		components.addAll(componentChain.getComponents());		
		return components;
	}
}
